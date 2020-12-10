from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import Layer, Sequential
from paddle.nn import Conv2D, Conv2DTranspose, ReLU, BatchNorm2D, GroupNorm, SyncBatchNorm
from paddle.nn.initializer import Normal, Constant, XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 norm_type='bn',
                 norm_name=None,
                 name=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn']

        self.conv = Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=1,
            weight_attr=ParamAttr(
                name=name + "_weight",
                initializer=Normal(mean=0., std=0.01),
                learning_rate=1.),
            bias_attr = ParamAttr(
                name=name + "_bias",
                initializer=Constant(value=0),
                learning_rate=2.))

        param_attr = ParamAttr(
            name=norm_name + "_scale",
            learning_rate=1.,
            regularizer=L2Decay(0.))
        bias_attr = ParamAttr(
            name=norm_name + "_offset",
            learning_rate=1.,
            regularizer=L2Decay(0.))
        if norm_type in ['bn', 'sync_bn']:
            self.norm = BatchNorm2D(
                ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = GroupNorm(
                num_groups=32,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class ScaleReg(nn.Layer):
    def __init__(self):
        super(ScaleReg, self).__init__()
        self.scale_reg = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=1.)),
            dtype="float32")

    def forward(self, inputs):
        out = inputs * self.scale_reg
        return out


@register
class FCOSFeat(nn.Layer):
    def __init__(self, feat_in=256, feat_out=256, num_convs=4, norm_type='bn'):
        super(FCOSFeat, self).__init__()
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.num_convs = num_convs
        self.norm_type = norm_type

        self.cls_subnet_convs = []
        self.reg_subnet_convs = []
        for i in range(self.num_convs):
            in_c = self.feat_in if i == 0 else self.feat_out

            cls_conv_name = 'fcos_head_cls_tower_conv_{}'.format(i)
            cls_conv = self.add_sublayer(
                cls_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=self.feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    norm_name=cls_conv_name + '_norm',
                    name=cls_conv_name))
            self.cls_subnet_convs.append(cls_conv)

            reg_conv_name = 'fcos_head_reg_tower_conv_{}'.format(i)
            reg_conv = self.add_sublayer(
                reg_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=self.feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    norm_name=reg_conv_name + '_norm',
                    name=reg_conv_name))
            self.reg_subnet_convs.append(reg_conv)

    def forward(self, fpn_feats):
        fcos_cls_feats = []
        fcos_reg_feats = []
        for feat in fpn_feats:
            cls_feat = feat
            reg_feat = feat
            for i in range(self.num_convs):
                cls_feat = F.relu(self.cls_subnet_convs[i](cls_feat))
                reg_feat = F.relu(self.reg_subnet_convs[i](reg_feat))
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
    """
    __inject__ = ['fcos_feat', 'fcos_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 fcos_feat,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 fcos_loss='FCOSLoss',
                 norm_reg_targets=True,
                 centerness_on_reg=True,  # FCOSv2
                 use_dcn_in_tower=False):
        super(FCOSHead, self).__init__()
        self.fcos_feat = fcos_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.fcos_loss = fcos_loss
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.use_dcn_in_tower = use_dcn_in_tower

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
                    initializer=Constant(value=bias_init_value))))  #
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

        self.scales_regs = []
        for i in range(len(self.fpn_stride)):
            lvl = int(math.log(int(self.fpn_stride[i]), 2))
            feat_name = 'p{}_feat'.format(lvl)
            scale_reg = self.add_sublayer(
                feat_name,
                ScaleReg())
            self.scales_regs.append(scale_reg)

    def forward(self, fpn_feats, spatial_scale, mode):
        cls_logits_list = []
        bboxes_reg_list = []
        centerness_list = []
        assert len(fpn_feats) == len(self.fpn_stride), "The size of fpn_feats is not equal to size of fpn_stride"
        fcos_cls_feats, fcos_reg_feats = self.fcos_feat(fpn_feats)

        for scale_reg, fpn_stride, fcos_cls_feat, fcos_reg_feat in zip(self.scales_regs, self.fpn_stride, fcos_cls_feats,
                                                                       fcos_reg_feats):
            cls_logits = self.fcos_head_cls[0](fcos_cls_feat)
            bbox_reg = self.fcos_head_reg[0](fcos_reg_feat)
            bbox_reg = scale_reg(bbox_reg)
            if self.centerness_on_reg:
                centerness = self.fcos_head_centerness[0](fcos_reg_feat)
            else:
                centerness = self.fcos_head_centerness[0](fcos_cls_feat)

            if self.norm_reg_targets:
                bbox_reg = F.relu(bbox_reg)
                if mode == 'infer':
                    bbox_reg = bbox_reg * fpn_stride
            else:
                bbox_reg = paddle.exp(bbox_reg)

            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)
            centerness_list.append(centerness)
        return cls_logits_list, bboxes_reg_list, centerness_list

    def get_loss(self, fcos_head_outs, tag_labels, tag_bboxes, tag_centerness):
        cls_logits, bboxes_reg, centerness = fcos_head_outs
        return self.fcos_loss(cls_logits, bboxes_reg, centerness, tag_labels, tag_bboxes, tag_centerness)

    def _compute_locations(self, fpn_feats):
        """
        Args:
            fpn_feats (list): List of Variables for FPN feature maps
        Return:
            Anchor points for each feature map pixel
        """
        locations = []
        for lvl, feature in enumerate(fpn_feats):
            shape_fm = paddle.shape(feature)
            shape_fm.stop_gradient = True
            h = shape_fm[2]
            w = shape_fm[3]
            fpn_stride = self.fpn_stride[lvl]
            shift_x = fluid.layers.range(
                0, w * fpn_stride, fpn_stride, dtype='float32')
            shift_y = fluid.layers.range(
                0, h * fpn_stride, fpn_stride, dtype='float32')
            shift_x = paddle.unsqueeze(shift_x, axis=0)
            shift_y = paddle.unsqueeze(shift_y, axis=1)
            shift_x = paddle.expand_as(
                shift_x, feature[0, 0, :, :])
            shift_y = paddle.expand_as(
                shift_y, feature[0, 0, :, :])
            shift_x.stop_gradient = True
            shift_y.stop_gradient = True
            shift_x = paddle.reshape(shift_x, shape=[-1])
            shift_y = paddle.reshape(shift_y, shape=[-1])
            location = paddle.stack(
                [shift_x, shift_y], axis=-1) + fpn_stride // 2
            location.stop_gradient = True
            locations.append(location)
        return locations

    def get_prediction(self, locations, fcos_head_outs):
        cls_logits, bboxes_reg, centerness = fcos_head_outs
        return locations, cls_logits, bboxes_reg, centerness
