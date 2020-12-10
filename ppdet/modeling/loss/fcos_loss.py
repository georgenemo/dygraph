# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from IPython import embed
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.modeling import ops
from ppdet.core.workspace import register

INF = 1e8
__all__ = ['FCOSLoss']


def flatten_tensor(inputs, channel_first=False):
    """
    Flatten a Tensor
    Args:
        inputs  (Variables): Input Tensor
        channel_first(bool): if true the dimension order of
            Tensor is [N, C, H, W], otherwise is [N, H, W, C]
    Return:
        input_channel_last (Variables): The flattened Tensor in channel_last style
    """
    if channel_first:
        input_channel_last = paddle.transpose(
            inputs, perm=[0, 2, 3, 1])
    else:
        input_channel_last = inputs
    output_channel_last = paddle.flatten(input_channel_last, start_axis=0, stop_axis=2)
    return output_channel_last


@register
class FCOSLoss(nn.Layer):
    """
    FCOSLoss
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type(str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights(float): weight for location loss
    """

    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="giou",
                 reg_weights=1.0):
        super(FCOSLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights

    def __iou_loss(self, pred, targets, positive_mask, weights=None):
        """
        Calculate the loss for location prediction
        Args:
            pred          (Variables): bounding boxes prediction
            targets       (Variables): targets for positive samples
            positive_mask (Variables): mask of positive samples
            weights       (Variables): weights for each positive samples
        Return:
            loss (Varialbes): location loss
        """
        plw = paddle.multiply(pred[:, 0], positive_mask)
        pth = paddle.multiply(pred[:, 1], positive_mask)
        prw = paddle.multiply(pred[:, 2], positive_mask)
        pbh = paddle.multiply(pred[:, 3], positive_mask)
        tlw = paddle.multiply(targets[:, 0], positive_mask)
        tth = paddle.multiply(targets[:, 1], positive_mask)
        trw = paddle.multiply(targets[:, 2], positive_mask)
        tbh = paddle.multiply(targets[:, 3], positive_mask)
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True
        area_target = (tlw + trw) * (tth + tbh)
        area_predict = (plw + prw) * (pth + pbh)
        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)
        clw = paddle.maximum(plw, tlw)
        crw = paddle.maximum(prw, trw)
        cth = paddle.maximum(pth, tth)
        cbh = paddle.maximum(pbh, tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
            area_predict + area_target - area_inter + 1.0)
        ious = paddle.multiply(ious, positive_mask)
        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - paddle.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        return loss

    def forward(self, cls_logits, bboxes_reg, centerness, tag_labels, tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): list of Variables, which is predicted
                score for all anchor points with shape [N, M, C]
            bboxes_reg (list): list of Variables, which is predicted
                offsets for all anchor points with shape [N, M, 4]
            centerness (list): list of Variables, which is predicted
                centerness for all anchor points with shape [N, M, 1]
            tag_labels (list): list of Variables, which is category
                targets for each anchor point
            tag_bboxes (list): list of Variables, which is bounding
                boxes targets for positive samples
            tag_center (list): list of Variables, which is centerness
                targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        cls_logits_flatten_list = []
        bboxes_reg_flatten_list = []
        centerness_flatten_list = []
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(
                flatten_tensor(cls_logits[lvl], True))
            bboxes_reg_flatten_list.append(
                flatten_tensor(bboxes_reg[lvl], True))
            centerness_flatten_list.append(
                flatten_tensor(centerness[lvl], True))

            tag_labels_flatten_list.append(
                flatten_tensor(tag_labels[lvl], False))
            tag_bboxes_flatten_list.append(
                flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(
                flatten_tensor(tag_center[lvl], False))

        cls_logits_flatten = paddle.concat(
            cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = paddle.concat(
            bboxes_reg_flatten_list, axis=0)
        centerness_flatten = paddle.concat(
            centerness_flatten_list, axis=0)

        tag_labels_flatten = paddle.concat(
            tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = paddle.concat(
            tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = paddle.concat(
            tag_center_flatten_list, axis=0)
        tag_labels_flatten.stop_gradient = True
        tag_bboxes_flatten.stop_gradient = True
        tag_center_flatten.stop_gradient = True

        mask_positive = tag_labels_flatten > 0
        mask_positive.stop_gradient = True
        mask_positive_float = paddle.cast(mask_positive, dtype="float32")
        mask_positive_float.stop_gradient = True
        num_positive_fp32 = paddle.sum(mask_positive_float)
        num_positive_int32 = paddle.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_fp32.stop_gradient = True
        num_positive_int32.stop_gradient = True
        normalize_sum = paddle.sum(tag_center_flatten)
        normalize_sum.stop_gradient = True
        normalize_sum = paddle.sum(mask_positive_float * normalize_sum)
        normalize_sum.stop_gradient = True

        # expand_onehot_labels
        '''
        label_channels = cls_logits_flatten.shape[-1]
        tag_labels_bin = np.zeros(shape=(tag_labels_flatten.shape[0], label_channels)) # tag_labels_flatten.new_full((tag_labels_flatten.shape[0], label_channels), 0)
        #inds = np.nonzero((tag_labels_flatten >= 0) & (tag_labels_flatten < label_channels), as_tuple=False).squeeze()
        inds = np.nonzero((tag_labels_flatten > 0)).squeeze()[:, 0]
        if inds.numel() > 0:
            tag_labels_bin[inds, tag_labels_flatten[inds]] = 1
        # embed()

        tag_labels_flatten_bin = paddle.to_tensor(tag_labels_bin.astype('float32'),
        cls_loss = F.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten_bin)
                    #, normalizer=paddle.sum(num_positive_int32))# reduction='mean')
        '''
        cls_loss = fluid.layers.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten, num_positive_int32) / num_positive_fp32
        #cls_loss = F.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten, num_positive_int32) / num_positive_fp32

        reg_loss = self.__iou_loss(bboxes_reg_flatten, tag_bboxes_flatten, mask_positive_float, tag_center_flatten)
        reg_loss = paddle.multiply(reg_loss, mask_positive_float) / normalize_sum

        #ctn_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
        #    x=centerness_flatten,
        #    label=tag_center_flatten) * mask_positive_float / num_positive_fp32
        ctn_loss = ops.sigmoid_cross_entropy_with_logits(centerness_flatten, tag_center_flatten, ignore_index=-1, normalize=True)
        ctn_loss = paddle.multiply(ctn_loss, mask_positive_float) / normalize_sum

        loss_all = {
            "loss_centerness": paddle.sum(ctn_loss),
            "loss_cls": paddle.sum(cls_loss),
            "loss_box": paddle.sum(reg_loss)
        }
        return loss_all
