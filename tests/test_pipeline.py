import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from network.modeling import MEDeepLabDualStreamBackbone
from network.utils import _MEDeepLabModelWrapper
from utils.modalities import index_images, load_inference_pair, read_mask, read_edge, read_terrain
from utils.Landslide import WHULandslideDataset
from utils.m2f_transforms import M2FCompose, M2FRandomCrop, M2FToTensor, M2FNormalize
from utils.loss import BoundaryDiceLoss, SegmentationCriterion
from utils.checkpoint import load_checkpoint, load_model_weights
from utils.inference import resolve_config
from utils.engine import build_optimizer, build_scheduler, validate, build_metrics, save_checkpoint, restore_from_checkpoint
from utils.seg_metrics_logger import SegMetricsLogger
from train import get_argparser, train_one_epoch, compute_training_losses


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.shape[0])
        feature = self.bn(self.conv(x)).relu()
        return {'out': feature, 'low_level': feature}


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg = nn.Conv2d(4, 2, 1)
        self.edge = nn.Conv2d(4, 1, 1)

    def forward(self, features):
        return {'seg': self.seg(features['out']), 'edge': self.edge(features['out'])}


def toy_model():
    return _MEDeepLabModelWrapper(MEDeepLabDualStreamBackbone(ToyBackbone(), ToyBackbone(), 4, 4), ToyHead())


def grad_size(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        self.model = toy_model()
        self.rgb, self.dem = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)

    def test_dem_requires_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, 'explicit has_dem'):
            self.model(self.rgb, self.dem)

    def test_nonzero_missing_skips_encoder_valid_black_uses_it(self):
        self.model.eval()
        flags = torch.tensor([False, True])
        self.dem[0].fill_(-2.1)
        self.dem[1].zero_()
        result = self.model(self.rgb, self.dem, has_dem=flags)
        self.assertEqual(self.model.backbone.dem_backbone.seen, [1])
        torch.testing.assert_close(result['used_dem'], flags)
        alternate = self.dem.clone()
        alternate[0].fill_(1000)
        changed = self.model(self.rgb, alternate, has_dem=flags)
        torch.testing.assert_close(result['seg'], changed['seg'])
        self.model(self.rgb)
        self.assertEqual(self.model.backbone.dem_backbone.seen, [1, 1])

    def test_real_route_task_gradients_reach_dem_and_fusion(self):
        self.model(self.rgb, self.dem, has_dem=torch.ones(2, dtype=torch.bool))['seg'].square().mean().backward()
        self.assertGreater(grad_size(self.model.backbone.dem_backbone), 0)
        self.assertGreater(grad_size(self.model.backbone.mef), 0)
        self.assertEqual(grad_size(self.model.backbone.fhm), 0)

    def test_dropped_real_routes_to_fhm_and_retains_teacher(self):
        flags = torch.ones(2, dtype=torch.bool)
        result = self.model(self.rgb, self.dem, has_dem=flags, dem_drop_mask=flags)
        self.assertFalse(result['used_dem'].any())
        self.assertGreater(float(result['feat_real'].detach().abs().sum()), 0)
        result['seg'].square().mean().backward()
        self.assertGreater(grad_size(self.model.backbone.fhm), 0)
        self.assertEqual(grad_size(self.model.backbone.dem_backbone), 0)

    def test_consistency_only_updates_fhm(self):
        result = self.model(self.rgb, self.dem, has_dem=torch.ones(2, dtype=torch.bool))
        nn.functional.l1_loss(result['feat_fake'], result['feat_real'].detach()).backward()
        self.assertGreater(grad_size(self.model.backbone.fhm), 0)
        self.assertEqual(grad_size(self.model.backbone.rgb_backbone), 0)
        self.assertEqual(grad_size(self.model.backbone.dem_backbone), 0)

    def test_changed_terrain_changes_real_route(self):
        self.model.eval()
        flags = torch.ones(2, dtype=torch.bool)
        real = self.model(self.rgb, self.dem, has_dem=flags)['seg']
        wrong = self.model(self.rgb, self.dem.flip([3]), has_dem=flags)['seg']
        self.assertGreater(float((real - wrong).detach().abs().max()), 1e-7)


class DataTests(unittest.TestCase):
    def test_png_and_missing_routes_match_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dem').mkdir()
            Image.new('RGB', (8, 8), (22, 100, 200)).save(root / 'sample.jpg')
            Image.new('RGB', (8, 8), (0, 0, 0)).save(root / 'dem' / 'sample.PNG')
            index = index_images(root / 'dem')
            _, rgb, dem, present, path = load_inference_pair(root / 'sample.jpg', index, 'required')
            self.assertTrue(present)
            self.assertEqual(rgb.shape, dem.shape)
            self.assertGreater(float(dem.abs().sum()), 0)
            _, _, missing, present, _ = load_inference_pair(root / 'sample.jpg', index, 'missing')
            self.assertFalse(present)
            self.assertEqual(float(missing.abs().sum()), 0)
            with self.assertRaises(FileNotFoundError):
                load_inference_pair(root / 'sample.jpg', {}, 'required')
            Image.new('L', (8, 8), 100).save(root / 'gray.png')
            with self.assertRaises(ValueError):
                read_terrain(root / 'gray.png')
            with self.assertRaises(ValueError):
                read_terrain(path, (9, 9))

    def test_palette_indices_edges_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = Image.fromarray(np.array([[0, 1], [255, 1]], dtype=np.uint8)).convert('P')
            palette = [0] * 768
            palette[3:6] = [255, 255, 255]
            mask.putpalette(palette)
            mask.save(root / 'mask.png')
            Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(root / 'edge.png')
            read = read_mask(root / 'mask.png')
            self.assertEqual(np.array(read)[0, 1], 1)
            edge = read_edge(root / 'edge.png')
            self.assertEqual(int(np.array(edge).sum()), 2)
            transform = M2FCompose([M2FRandomCrop(4), M2FToTensor(), M2FNormalize()])
            _, _, labels, edges = transform(Image.new('RGB', (2, 2)), Image.new('RGB', (2, 2)), read, edge)
            self.assertEqual(int((edges == 1).sum()), 2)
            self.assertEqual(int((edges == 255).sum()), 12)
            self.assertEqual(int((labels == 255).sum()), 13)


class IntegrationTests(unittest.TestCase):
    def test_training_validation_logging_and_resume(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        opts = get_argparser().parse_args([])
        opts.total_itrs, opts.epochs, opts.batch_size = 4, 2, 2
        opts.num_workers, opts.eval_missing_dem = 0, True
        opts.dem_warmup_epochs, opts.dem_dropout = 0, .5
        model = toy_model()
        loader = DataLoader(TensorDataset(torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8),
                                          torch.randint(0, 2, (4, 8, 8)), torch.randint(0, 2, (4, 1, 8, 8)).float(),
                                          torch.tensor([True, True, True, False])), batch_size=2)
        optimizer = build_optimizer(opts, model)
        scheduler = build_scheduler(opts, optimizer)
        criterion = SegmentationCriterion()
        diagnostics = train_one_epoch(opts, model, loader, torch.device('cpu'), optimizer, scheduler, criterion, 0)
        self.assertEqual(diagnostics['route_real'] + diagnostics['route_fhm'], 4)
        self.assertEqual(diagnostics['dem_present'], 3)
        score = validate(opts, model, loader, torch.device('cpu'), build_metrics(opts), criterion)
        self.assertEqual(score['Paired Samples'], 3)
        with tempfile.TemporaryDirectory() as directory:
            logger = SegMetricsLogger(directory)
            logger.log_epoch(1, opts.lr, diagnostics['train_loss'], diagnostics['train_pacc'], score['Loss'], score, diagnostics)
            logger.save_plots()
            with open(Path(directory) / 'metrics.csv', encoding='utf-8') as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(int(row['dem_present']), 3)
                self.assertNotEqual(row['paired_missing_miou'], '')
            path = str(Path(directory) / 'test.pth')
            save_checkpoint(path, model, optimizer, scheduler, 2, .25, opts.model, config=vars(opts), epoch=1)
            restored = load_checkpoint(path)
            self.assertEqual(restored['epoch'], 1)
            fresh = toy_model()
            load_model_weights(fresh, restored)
            opts.ckpt, opts.continue_training = path, True
            fresh_optimizer = build_optimizer(opts, fresh)
            fresh_scheduler = build_scheduler(opts, fresh_optimizer)
            _, best, iters, epoch = restore_from_checkpoint(opts, fresh, fresh_optimizer, fresh_scheduler, loader)
            self.assertEqual((best, iters, epoch), (.25, 2, 1))
            self.assertEqual(fresh_scheduler.last_epoch, scheduler.last_epoch)

    def test_loss_weights_applied_once(self):
        opts = get_argparser().parse_args([])
        outputs = {'seg': torch.randn(2, 2, 4, 4), 'edge': torch.randn(2, 1, 4, 4)}
        labels, edges = torch.randint(0, 2, (2, 4, 4)), torch.randint(0, 2, (2, 1, 4, 4)).float()
        total, parts = compute_training_losses(opts, outputs, labels, edges, torch.zeros(2, dtype=torch.bool),
                                                SegmentationCriterion(), BoundaryDiceLoss())
        torch.testing.assert_close(total, parts['loss_seg'] + opts.alpha * parts['loss_edge'])

    def test_inference_rejects_incompatible_stride(self):
        from M2FPredict_New import get_argparser as inference_parser
        opts = inference_parser().parse_args(['--ckpt', 'unused', '--output_stride', '8'])
        checkpoint = {'config': {'model': 'm2f_deeplabv3plus_xception16', 'num_classes': 2,
                                  'output_stride': 16, 'separable_conv': False}}
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            resolve_config(checkpoint, opts)


if __name__ == '__main__':
    unittest.main()
