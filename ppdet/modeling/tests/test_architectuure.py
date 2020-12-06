from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import paddle
import paddle.fluid as fluid
import os
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)
from ppdet.modeling.tests.decorator_helper import prog_scope
from ppdet.core.workspace import load_config, merge_config, create
# from ppdet.utils.check import enable_static_mode

class TestFCOS(unittest.TestCase):
    def setUp(self):
        self.set_config()
        self.cfg = load_config(self.cfg_file)
        self.detector_type = self.cfg['architecture']

    def set_config(self):
        self.cfg_file = 'configs/fcos_r50_fpn_coco_debug.yml'

    @prog_scope()
    def test_train(self):
        model = create(self.detector_type)
        inputs_def = self.cfg['TrainReader']['inputs_def']
        inputs_def['image_shape'] = [3, None, None]
        feed_vars, _ = model.build_inputs(**inputs_def)
        train_fetches = model.train(feed_vars)

    @prog_scope()
    def test_test(self):
        inputs_def = self.cfg['EvalReader']['inputs_def']
        inputs_def['image_shape'] = [3, None, None]
        model = create(self.detector_type)
        feed_vars, _ = model.build_inputs(**inputs_def)
        test_fetches = model.eval(feed_vars)

if __name__ == '__main__':
    # enable_static_mode()
    unittest.main()
