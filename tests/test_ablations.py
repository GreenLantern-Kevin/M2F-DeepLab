import unittest

import torch

from network import modeling
from network._deeplab import DeepLabHeadV3Plus, MEDeepLabHeadV3Plus
from network.utils import _MEDeepLabModelWrapper
from utils.inference import missing_route, resolve_config
from utils.engine import build_model
from test_pipeline import ToyBackbone, ToyHead, grad_size
from train import get_argparser
from M2FPredict_New import get_argparser as prediction_parser


torch.set_num_threads(2)


class AblationTests(unittest.TestCase):
    def test_b_physically_has_no_fhm_and_missing_features_are_zero(self):
        model = modeling.MEDeepLabDualStreamBackbone(ToyBackbone(), ToyBackbone(), 4, 4,
                                                    use_fhm=False, fusion='concat').eval()
        self.assertIsNone(model.fhm)
        self.assertIsNone(model.mef)
        self.assertFalse(any('fhm' in key or 'mef' in key for key in model.state_dict()))
        rgb, dem = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
        selected = []
        hook = model.concat.register_forward_pre_hook(lambda module, args: selected.append(args[1].detach()))
        result = model(rgb, dem, has_dem=torch.tensor([True, True]), dem_drop_mask=torch.tensor([False, True]))
        hook.remove()
        self.assertIsNone(result['feat_fake'])
        self.assertFalse(result['used_fhm'].any())
        self.assertGreater(float(selected[0][0].abs().sum()), 0)
        self.assertEqual(float(selected[0][1].abs().sum()), 0)
        result['out'].square().mean().backward()
        self.assertGreater(grad_size(model.dem_backbone), 0)
        self.assertGreater(grad_size(model.concat), 0)

    def test_c_and_d_use_fhm_when_dropped_and_keep_real_teacher(self):
        for fusion in ('concat', 'mef'):
            with self.subTest(fusion=fusion):
                model = modeling.MEDeepLabDualStreamBackbone(ToyBackbone(), ToyBackbone(), 4, 4,
                                                            use_fhm=True, fusion=fusion)
                rgb, dem = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
                flags = torch.ones(2, dtype=torch.bool)
                result = model(rgb, dem, has_dem=flags, dem_drop_mask=flags)
                self.assertTrue(result['used_fhm'].all())
                self.assertGreater(float(result['feat_real'].detach().abs().sum()), 0)
                result['out'].square().mean().backward()
                self.assertGreater(grad_size(model.fhm), 0)
                self.assertEqual(grad_size(model.dem_backbone), 0)

    def test_real_dem_logits_do_not_depend_on_fhm_weights(self):
        model = _MEDeepLabModelWrapper(
            modeling.MEDeepLabDualStreamBackbone(ToyBackbone(), ToyBackbone(), 4, 4), ToyHead()).eval()
        rgb, dem, flags = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8), torch.ones(2, dtype=torch.bool)
        with torch.no_grad():
            before = model(rgb, dem, has_dem=flags)['seg']
            for parameter in model.backbone.fhm.parameters():
                parameter.add_(1)
            after = model(rgb, dem, has_dem=flags)['seg']
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_b_and_c_same_shared_weights_produce_same_full_dem_features(self):
        # With no dropout, adding a detached FHM does not alter the real-DEM inference path.
        b = modeling.MEDeepLabDualStreamBackbone(ToyBackbone(), ToyBackbone(), 4, 4, use_fhm=False, fusion='concat').eval()
        c = modeling.MEDeepLabDualStreamBackbone(ToyBackbone(), ToyBackbone(), 4, 4, use_fhm=True, fusion='concat').eval()
        c.load_state_dict(b.state_dict(), strict=False)
        rgb, dem, flags = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8), torch.ones(2, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(b(rgb, dem, has_dem=flags)['out'], c(rgb, dem, has_dem=flags)['out'], rtol=0, atol=0)

    def test_factories_and_saved_configs_select_actual_structures(self):
        # Meta construction checks full 2048-channel layouts without allocating large weights.
        for letter, use_fhm, use_mef in [('b', False, False), ('c', True, False), ('d', True, True)]:
            name = 'm2f_deeplabv3plus_xception16_' + letter
            with self.subTest(model=name), torch.device('meta'):
                opts = get_argparser().parse_args(['--model', name, '--no-pretrained_backbone'])
                model = build_model(opts)
                self.assertEqual(model.backbone.fhm is not None, use_fhm)
                self.assertEqual(model.backbone.mef is not None, use_mef)
                self.assertIsInstance(model.classifier, DeepLabHeadV3Plus)
                self.assertFalse(hasattr(model.classifier, 'edge_classifier'))
                self.assertEqual(model.classifier.classifier[0].in_channels, 304)
                config = resolve_config({'config': vars(opts)}, prediction_parser().parse_args(['--ckpt', 'unused']))
                self.assertEqual(config['model'], name)
                self.assertEqual(missing_route(config), 'zero_features' if letter == 'b' else 'fhm')
        with torch.device('meta'):
            full = modeling.m2f_deeplabv3plus_xception16(num_classes=2, output_stride=16, pretrained_backbone=False)
            self.assertIsInstance(full.classifier, MEDeepLabHeadV3Plus)
            self.assertEqual(full.classifier.classifier[0].in_channels, 305)



if __name__ == '__main__':
    unittest.main()
