from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid
from ppdet.modeling.ops import multiclass_nms

import paddle
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Layer, Sequential
from paddle.nn import Conv2D, Conv2DTranspose, ReLU
from paddle.nn.initializer import Normal, Constant, NumpyArrayInitializer
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops


@register
class FCOSFeat(nn.Layer):
    def __init__(self, feat_in=256, feat_out=256, num_convs=4, norm_type='gn'):
        super(FCOSFeat, self).__init__()
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.num_convs = num_convs
        self.norm_type = norm_type

        self.cls_subnet_convs = []
        self.reg_subnet_convs = []
        for i in range(self.num_convs):
            chn = self.feat_in if i == 0 else self.feat_out

            cls_conv_name = 'fcos_head_cls_tower_conv_{}'.format(i)
            cls_conv = self.add_sublayer(
                cls_conv_name,
                Conv2D(
                    in_channels=chn,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=in_c)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            self.cls_subnet_convs.append(cls_conv)

            reg_conv_name = 'fcos_head_reg_tower_conv_{}'.format(i)
            chn = self.feat_in if i == 0 else self.feat_out
            reg_conv = self.add_sublayer(
                reg_conv_name,
                Conv2D(
                    in_channels=chn,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=in_c)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            self.reg_subnet_convs.append(reg_conv)

    def forward(self, fpn_feats):
        fcos_cls_feats = []
        fcos_reg_feats = []
        for feat in fpn_feats:
            cls_feat = feat
            reg_feat = feat
            for cls_conv in self.cls_subnet_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_subnet_convs:
                reg_feat = reg_conv(reg_feat)
            fcos_cls_feats.append(cls_feat)
            fcos_reg_feats.append(reg_feat)
        return fcos_cls_feats, fcos_reg_feats

@register
class FCOSHead(nn.Layer):
    """
    FCOSHead
    Args:
        num_classes       (int): Number of classes
        fpn_stride       (list): The stride of each FPN Layer
        prior_prob      (float): Used to set the bias init for the class prediction layer
        fcos_loss      (object): Instance of 'FCOSLoss'
        norm_reg_targets (bool): Normalization the regression target if true
        centerness_on_reg(bool): The prediction of centerness on regression or clssification branch
        use_dcn_in_tower (bool): Ues deformable conv on FCOSHead if true
        nms            (object): Instance of 'MultiClassNMS'
    """
    __inject__ = ['fcos_feat', 'fcos_loss', 'nms']
    __shared__ = ['num_classes']

    def __init__(self,
                 fcos_feat,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 fcos_loss='FCOSLoss',
                 norm_reg_targets=True,
                 centerness_on_reg=True, # FCOSv2
                 use_dcn_in_tower=False):
        self.fcos_feat = fcos_feat
        if isinstance(fcos_feat, dict):
            self.fcos_feat = FCOSFeat(**fcos_feat)

        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.fcos_loss = fcos_loss
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.use_dcn_in_tower = use_dcn_in_tower
        # self.nms = MultiClassNMS(**nms)
        self.batch_size = 8 #

        self.fcos_head_cls = []
        self.fcos_head_reg = []
        self.fcos_head_centerness = []

        conv_cls_name = "fcos_head_cls"
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        fcos_head_cls = self.add_sublayer(
            conv_cls_name,
            Conv2D(
                in_channels=256,
                out_channels=self.num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(
                    name=conv_cls_name + "_weights",
                    initializer=Normal(mean=0., std=0.01)),
                bias_attr=ParamAttr(
                    name=conv_cls_name + "_bias",
                    initializer=Constant(value=bias_init_value)))) #
        self.fcos_head_cls.append(fcos_head_cls)

        conv_reg_name = "fcos_head_reg"
        fcos_head_reg = self.add_sublayer(
            conv_reg_name,
            Conv2D(
                in_channels=256,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(
                    name=conv_reg_name + "_weights",
                    initializer=Normal(mean=0., std=0.01)),
                bias_attr=ParamAttr(
                    name=conv_reg_name + "_bias",
                    initializer=Constant(value=0))))
        self.fcos_head_reg.append(fcos_head_reg)

        conv_centerness_name = "fcos_head_centerness"
        fcos_head_centerness = self.add_sublayer(
            conv_centerness_name,
            Conv2D(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(
                    name=conv_centerness_name + "_weights",
                    initializer=Normal(mean=0., std=0.01)),
                bias_attr=ParamAttr(
                    name=conv_centerness_name + "_bias",
                    initializer=Constant(value=0))))
        self.fcos_head_centerness.append(fcos_head_centerness)


    def forward(self, feats):
        cls_logits_list = []
        bboxes_reg_list = []
        centerness_list = []
        assert len(feats) == len(self.fpn_stride), "The size of fpn_feats is not equal to size of fpn_stride"
        fcos_cls_feats, fcos_reg_feats = FCOSFeat(**feats)

        for fpn_name, fpn_stride, fcos_cls_feat, fcos_reg_feat in zip(body_feats, self.fpn_stride, fcos_cls_feats, fcos_reg_feats):
            cls_logits = self.fcos_head_cls[0](fcos_cls_feat)
            bbox_reg = self.fcos_head_reg[0](fcos_reg_feat)
            if self.centerness_on_reg:
                centerness = self.fcos_head_centerness[0](fcos_reg_feat)
            else:
                centerness = self.fcos_head_centerness[0](fcos_cls_feat)
            ###
            scale = fluid.layers.create_parameter(
                shape=[1, ],
                dtype="float32",
                name="%s_scale_on_reg" % fpn_name,
                default_initializer=fluid.initializer.Constant(1.))
            bbox_reg = bbox_reg * scale
            if self.norm_reg_targets:
                bbox_reg = fluid.layers.relu(bbox_reg)
                if self.inputs['mode'] == 'infer':
                    bbox_reg = bbox_reg * self.fpn_stride
            else:
                bbox_reg = fluid.layers.exp(bbox_reg)

            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)
            centerness_list.append(centerness)
        return cls_logits_list, bboxes_reg_list, centerness_list


    def _get_output(self, body_feats, is_training=False):
        """
        Args:
            body_feates (list): the list of fpn feature maps
            is_training (bool): whether is train or test mode
        Return:
            cls_logits (Variables): prediction for classification
            bboxes_reg (Variables): prediction for bounding box
            centerness (Variables): prediction for ceterness
        """
        cls_logits = []
        bboxes_reg = []
        centerness = []
        assert len(body_feats) == len(self.fpn_stride), \
            "The size of body_feats is not equal to size of fpn_stride"
        for fpn_name, fpn_stride in zip(body_feats, self.fpn_stride):
            features = body_feats[fpn_name]
            scale = fluid.layers.create_parameter(
                shape=[1, ],
                dtype="float32",
                name="%s_scale_on_reg" % fpn_name,
                default_initializer=fluid.initializer.Constant(1.))
            cls_pred, bbox_pred, ctn_pred = self._fcos_head(
                features, fpn_stride, scale, is_training=is_training)
            cls_logits.append(cls_pred)
            bboxes_reg.append(bbox_pred)
            centerness.append(ctn_pred)
        return cls_logits, bboxes_reg, centerness

    def _compute_locations(self, features):
        """
        Args:
            features (list): List of Variables for FPN feature maps
        Return:
            Anchor points for each feature map pixel
        """
        locations = []
        for lvl, fpn_name in enumerate(features):
            feature = features[fpn_name]
            shape_fm = fluid.layers.shape(feature)
            shape_fm.stop_gradient = True
            h = shape_fm[2]
            w = shape_fm[3]
            fpn_stride = self.fpn_stride[lvl]
            shift_x = fluid.layers.range(
                0, w * fpn_stride, fpn_stride, dtype='float32')
            shift_y = fluid.layers.range(
                0, h * fpn_stride, fpn_stride, dtype='float32')
            shift_x = fluid.layers.unsqueeze(shift_x, axes=[0])
            shift_y = fluid.layers.unsqueeze(shift_y, axes=[1])
            shift_x = fluid.layers.expand_as(
                shift_x, target_tensor=feature[0, 0, :, :])
            shift_y = fluid.layers.expand_as(
                shift_y, target_tensor=feature[0, 0, :, :])
            shift_x.stop_gradient = True
            shift_y.stop_gradient = True
            shift_x = fluid.layers.reshape(shift_x, shape=[-1])
            shift_y = fluid.layers.reshape(shift_y, shape=[-1])
            location = fluid.layers.stack(
                [shift_x, shift_y], axis=-1) + fpn_stride // 2
            location.stop_gradient = True
            locations.append(location)
        return locations

    def __merge_hw(self, input, ch_type="channel_first"):
        """
        Args:
            input (Variables): Feature map whose H and W will be merged into one dimension
            ch_type     (str): channel_first / channel_last
        Return:
            new_shape (Variables): The new shape after h and w merged into one dimension
        """
        shape_ = fluid.layers.shape(input)
        bs = shape_[0]
        ch = shape_[1]
        hi = shape_[2]
        wi = shape_[3]
        img_size = hi * wi
        img_size.stop_gradient = True
        if ch_type == "channel_first":
            new_shape = fluid.layers.concat([bs, ch, img_size])
        elif ch_type == "channel_last":
            new_shape = fluid.layers.concat([bs, img_size, ch])
        else:
            raise KeyError("Wrong ch_type %s" % ch_type)
        new_shape.stop_gradient = True
        return new_shape

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn, im_info):
        """
        Args:
            locations (Variables): anchor points for current layer
            box_cls   (Variables): categories prediction
            box_reg   (Variables): bounding box prediction
            box_ctn   (Variables): centerness prediction
            im_info   (Variables): [h, w, scale] for input images
        Return:
            box_cls_ch_last  (Variables): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Variables): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        act_shape_cls = self.__merge_hw(box_cls)
        box_cls_ch_last = fluid.layers.reshape(
            x=box_cls,
            shape=[self.batch_size, self.num_classes, -1],
            actual_shape=act_shape_cls)
        box_cls_ch_last = fluid.layers.sigmoid(box_cls_ch_last)
        act_shape_reg = self.__merge_hw(box_reg, "channel_last")
        box_reg_ch_last = fluid.layers.transpose(box_reg, perm=[0, 2, 3, 1])
        box_reg_ch_last = fluid.layers.reshape(
            x=box_reg_ch_last,
            shape=[self.batch_size, -1, 4],
            actual_shape=act_shape_reg)
        act_shape_ctn = self.__merge_hw(box_ctn)
        box_ctn_ch_last = fluid.layers.reshape(
            x=box_ctn,
            shape=[self.batch_size, 1, -1],
            actual_shape=act_shape_ctn)
        box_ctn_ch_last = fluid.layers.sigmoid(box_ctn_ch_last)

        box_reg_decoding = fluid.layers.stack(
            [
                locations[:, 0] - box_reg_ch_last[:, :, 0],
                locations[:, 1] - box_reg_ch_last[:, :, 1],
                locations[:, 0] + box_reg_ch_last[:, :, 2],
                locations[:, 1] + box_reg_ch_last[:, :, 3]
            ],
            axis=1)
        box_reg_decoding = fluid.layers.transpose(
            box_reg_decoding, perm=[0, 2, 1])
        # recover the location to original image
        im_scale = im_info[:, 2]
        box_reg_decoding = box_reg_decoding / im_scale
        box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last
        return box_cls_ch_last, box_reg_decoding

    def _post_processing(self, locations, cls_logits, bboxes_reg, centerness, im_info):
        """
        Args:
            locations   (list): List of Variables composed by center of each anchor point
            cls_logits  (list): List of Variables for class prediction
            bboxes_reg  (list): List of Variables for bounding box prediction
            centerness  (list): List of Variables for centerness prediction
            im_info(Variables): [h, w, scale] for input images
        Return:
            pred (LoDTensor): predicted bounding box after nms,
                the shape is n x 6, last dimension is [label, score, xmin, ymin, xmax, ymax]
        """
        pred_boxes_ = []
        pred_scores_ = []
        for _, (
                pts, cls, box, ctn
        ) in enumerate(zip(locations, cls_logits, bboxes_reg, centerness)):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, im_info)
            pred_boxes_.append(pred_boxes_lvl)
            pred_scores_.append(pred_scores_lvl)
        pred_boxes = fluid.layers.concat(pred_boxes_, axis=1)
        pred_scores = fluid.layers.concat(pred_scores_, axis=2)
        pred = multiclass_nms(pred_boxes, pred_scores, score_threshold=0.025,
                              nms_top_k=1000, keep_top_k=100,
                              nms_threshold=0.6, background_label=-1) ###
        return pred

    def get_loss(self, input, tag_labels, tag_bboxes, tag_centerness):
        """
        Calculate the loss for FCOS
        Args:
            input               (list): List of Variables for feature maps from FPN layers
            tag_labels     (Variables): category targets for each anchor point
            tag_bboxes     (Variables): bounding boxes  targets for positive samples
            tag_centerness (Variables): centerness targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
                regression loss and centerness regression loss
        """
        cls_logits, bboxes_reg, centerness = self._get_output(
            input, is_training=True)
        loss = self.fcos_loss(cls_logits, bboxes_reg, centerness, tag_labels,
                              tag_bboxes, tag_centerness)
        return loss

    def get_prediction(self, fpn_inputs, im_info):
        """
        Decode the prediction
        Args:
            fpn_inputs  (list): List of Variables for feature maps from FPN layers
            im_info(Variables): [h, w, scale] for input images
        Return:
            the bounding box prediction
        """
        cls_logits, bboxes_reg, centerness = self._get_output(
            fpn_inputs, is_training=False)
        locations = self._compute_locations(fpn_inputs)
        pred = self._post_processing(locations, cls_logits, bboxes_reg, centerness, im_info)
        return {"bbox": pred}
