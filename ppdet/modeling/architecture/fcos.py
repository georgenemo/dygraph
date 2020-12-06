from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from paddle import fluid
import paddle
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['FCOS']


@register
class FCOS(BaseArch):
    """
    FCOS architecture, see https://arxiv.org/abs/1904.01355

    Args:
        backbone (object): backbone instance
        neck (object): feature pyramid network instance
        fcos_head (object): `FCOSHead` instance
    """
    __category__ = 'architecture'
    __inject__ = [
        'backbone',
        'neck',
        'fcos_head',
        'fcos_post_process',
    ]

    def __init__(self,
                 backbone,
                 neck,
                 fcos_head='FCOSHead',
                 fcos_post_process='FCOSPostProcess'):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fcos_head = fcos_head
        self.fcos_post_process = fcos_post_process

    def _inputs_def(self, image_shape, fields):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image': {'shape': im_shape, 'dtype': 'float32', 'lod_level': 0},
            'im_shape': {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_info': {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_id': {'shape': [None, 1], 'dtype': 'int64', 'lod_level': 0},
            'gt_bbox': {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 1},
            'gt_score': {'shape': [None, 1], 'dtype': 'float32', 'lod_level': 1},
            'is_crowd': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 1},
            'is_difficult': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 1}
        }
        # yapf: disable
        if 'fcos_target' in fields:
            targets_def = {
                'labels0': {'shape': [None, None, None, 1], 'dtype': 'int32', 'lod_level': 0},
                'reg_target0': {'shape': [None, None, None, 4], 'dtype': 'float32', 'lod_level': 0},
                'centerness0': {'shape': [None, None, None, 1], 'dtype': 'float32', 'lod_level': 0},
                'labels1': {'shape': [None, None, None, 1], 'dtype': 'int32', 'lod_level': 0},
                'reg_target1': {'shape': [None, None, None, 4], 'dtype': 'float32', 'lod_level': 0},
                'centerness1': {'shape': [None, None, None, 1], 'dtype': 'float32', 'lod_level': 0},
                'labels2': {'shape': [None, None, None, 1], 'dtype': 'int32', 'lod_level': 0},
                'reg_target2': {'shape': [None, None, None, 4], 'dtype': 'float32', 'lod_level': 0},
                'centerness2': {'shape': [None, None, None, 1], 'dtype': 'float32', 'lod_level': 0},
                'labels3': {'shape': [None, None, None, 1], 'dtype': 'int32', 'lod_level': 0},
                'reg_target3': {'shape': [None, None, None, 4], 'dtype': 'float32', 'lod_level': 0},
                'centerness3': {'shape': [None, None, None, 1], 'dtype': 'float32', 'lod_level': 0},
                'labels4': {'shape': [None, None, None, 1], 'dtype': 'int32', 'lod_level': 0},
                'reg_target4': {'shape': [None, None, None, 4], 'dtype': 'float32', 'lod_level': 0},
                'centerness4': {'shape': [None, None, None, 1], 'dtype': 'float32', 'lod_level': 0},
            }
            # yapf: enable

            # downsample = 128
            for k, stride in enumerate(self.fcos_head.fpn_stride):
                k_lbl = 'labels{}'.format(k)
                k_box = 'reg_target{}'.format(k)
                k_ctn = 'centerness{}'.format(k)
                grid_y = image_shape[-2] // stride if image_shape[-2] else None
                grid_x = image_shape[-1] // stride if image_shape[-1] else None
                if grid_x is not None:
                    num_pts = grid_x * grid_y
                    num_dim2 = 1
                else:
                    num_pts = None
                    num_dim2 = None
                targets_def[k_lbl]['shape'][1] = num_pts
                targets_def[k_box]['shape'][1] = num_pts
                targets_def[k_ctn]['shape'][1] = num_pts
                targets_def[k_lbl]['shape'][2] = num_dim2
                targets_def[k_box]['shape'][2] = num_dim2
                targets_def[k_ctn]['shape'][2] = num_dim2
            inputs_def.update(targets_def)
        return inputs_def

    def build_inputs(self, data, input_def):
        image_shape = [3, None, None]
        # use_dataloader=True
        # iterable=False
        inputs_def = self._inputs_def(image_shape, input_def.fields)
        if "fcos_target" in input_def.fields:  # for-train
            for i in range(len(self.fcos_head.fpn_stride)):
                input_def.fields.extend(
                    ['labels%d' % i, 'reg_target%d' % i, 'centerness%d' % i])
            input_def.fields.remove('fcos_target')
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in input_def.fields])
        # loader = fluid.io.DataLoader.from_generator(
        #     feed_list=list(feed_vars.values()),
        #     capacity=16,
        #     use_double_buffer=True,
        #     iterable=iterable) if use_dataloader else None
        # return feed_vars, loader
        return feed_vars  # self.inputs

    def model_arch(self, ):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        fpn_feats, spatial_scale = self.neck(body_feats)

        # FCOS_head
        self.fcos_head_outs = self.fcos_head(fpn_feats, spatial_scale)

        if self.inputs['mode'] == 'infer':
            locations = self.fcos_head._compute_locations(fpn_feats)
            locations, cls_logits, bboxes_reg, centerness = self.fcos_head.get_prediction(locations,
                                                                                          self.fcos_head_outs)
            self.bboxes = self.fcos_post_process(locations, cls_logits, bboxes_reg, centerness, self.inputs['im_info'])

    def get_loss(self, ):
        loss = {}
        tag_labels = []
        tag_bboxes = []
        tag_centerness = []
        for i in range(len(self.fcos_head.fpn_stride)):
            # reg_target, labels, scores, centerness
            k_lbl = 'labels{}'.format(i)
            if k_lbl in self.inputs:
                tag_labels.append(self.inputs[k_lbl])
            k_box = 'reg_target{}'.format(i)
            if k_box in self.inputs:
                tag_bboxes.append(self.inputs[k_box])
            k_ctn = 'centerness{}'.format(i)
            if k_ctn in self.inputs:
                tag_centerness.append(self.inputs[k_ctn])
        loss_fcos = self.fcos_head.get_loss(self.fcos_head_outs, tag_labels, tag_bboxes, tag_centerness)
        loss.update(loss_fcos)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self, return_numpy=True):
        bbox, bbox_num = self.bboxes
        if return_numpy:
            outs = {
                'bbox': bbox.numpy(),
                'bbox_num': bbox_num.numpy(),
                'im_id': self.inputs['im_id'].numpy()
            }
        else:
            outs = [bbox, bbox_num]
        return outs
