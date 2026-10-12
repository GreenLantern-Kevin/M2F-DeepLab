"""Checkpoint/configuration and input contract shared by prediction and CAM."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import torch
import network
from .checkpoint import load_checkpoint, load_model_weights, read_model_config
from .modalities import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_MODEL_CONFIG = {'model': 'm2f_deeplabv3plus_xception16', 'num_classes': 2,
                 'output_stride': 16, 'separable_conv': False}


def add_common_arguments(parser, default_dem_mode='auto'):
    parser.add_argument('--ckpt', required=True, help='Explicit checkpoint shared by real/missing comparisons')
    parser.add_argument('--input', default='BijieLandslide/VOC2012/JPEGImages')
    parser.add_argument('--dem_dir', default='BijieLandslide/VOC2012/DEMImages')
    parser.add_argument('--dem_mode', choices=('auto', 'required', 'missing'), default=default_dem_mode)
    parser.add_argument('--model', choices=network.modeling.MODEL_NAMES, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--output_stride', type=int, choices=(8, 16), default=None)
    parser.add_argument('--separable_conv', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--resize', type=int, default=None, help='Joint square size; 0 keeps original size. Uses checkpoint eval_resize when omitted.')
    parser.add_argument('--mask_encoding', choices=('indices', 'binary255'), default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output_dir', default=None, help='Parent directory; each invocation creates a separate run folder')
    return parser


def resolve_config(checkpoint, opts):
    stored = read_model_config(checkpoint)
    has_config = bool(checkpoint.get('config'))
    resolved = dict(stored)
    for name, default in DEFAULT_MODEL_CONFIG.items():
        explicit = getattr(opts, name, None)
        if name in stored and explicit is not None and explicit != stored[name]:
            raise ValueError(f'--{name}={explicit!r} conflicts with checkpoint {stored[name]!r}')
        if has_config and name not in stored and explicit is None:
            raise ValueError(f'Checkpoint config is missing {name}; specify --{name} explicitly')
        resolved[name] = stored.get(name, default) if explicit is None else explicit
    if not has_config:
        warnings.warn('Checkpoint has no config. Model defaults (unless explicitly supplied): '
                      f'{ {key: resolved[key] for key in DEFAULT_MODEL_CONFIG} }. '
                      'Verify --output_stride and --separable_conv against training; strict tensor loading '
                      'cannot detect a stride mismatch. Evaluation defaults: resize=256.', RuntimeWarning)
    if resolved['num_classes'] != 2:
        raise ValueError('This landslide prediction/CAM workflow expects two classes.')
    eval_defaults = {'resize': stored.get('eval_resize', stored.get('val_size', 256)),
                     'mask_encoding': stored.get('mask_encoding', 'indices')}
    for name, fallback in eval_defaults.items():
        explicit = getattr(opts, name, None)
        resolved[name] = fallback if explicit is None else explicit
    if resolved['resize'] is not None and resolved['resize'] < 0:
        raise ValueError('--resize must be nonnegative')
    resolved['resize'] = resolved['resize'] or None
    resolved['normalization'] = {'mean': list(IMAGENET_MEAN), 'std': list(IMAGENET_STD),
                                  'rgb_and_dem': True, 'terrain_channels': ['elevation', 'slope', 'aspect']}
    return resolved


def load_inference_model(opts):
    checkpoint = load_checkpoint(opts.ckpt, map_location='cpu')
    config = resolve_config(checkpoint, opts)
    model = network.modeling.__dict__[config['model']](
        num_classes=config['num_classes'], output_stride=config['output_stride'], pretrained_backbone=False)
    if config['separable_conv'] and 'plus' in config['model']:
        network.convert_to_separable_conv(model.classifier)
    report = load_model_weights(model, checkpoint)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if opts.device == 'auto' else opts.device)
    model.to(device).eval()
    return model, config, device, report


def is_dual_model(config):
    return config['model'] in network.modeling.DUAL_MODEL_NAMES


def missing_route(config):
    if not is_dual_model(config):
        return 'rgb'
    return 'zero_features' if config['model'] in network.modeling.ZERO_FALLBACK_MODEL_NAMES else 'fhm'


def forward_outputs(model, config, rgb, dem, has_dem):
    output = model(rgb, dem, has_dem=has_dem) if is_dual_model(config) else model(rgb)
    return output if isinstance(output, dict) else {'seg': output}


def evaluation_options(config):
    return SimpleNamespace(**{key: config[key] for key in ('model',)})


def create_run_directory(base, kind, mode):
    root = Path(base or ('run/' + kind))
    run_dir = root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + mode)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(run_dir, opts, config, report, records, extra=None):
    manifest = {'checkpoint': str(Path(opts.ckpt).resolve()), 'resolved_config': config,
                'arguments': vars(opts), 'checkpoint_loading': report,
                'dem_summary': {'images': len(records),
                                'real_dem': sum(row['has_dem'] for row in records),
                                'missing_dem': sum(not row['has_dem'] for row in records)},
                'images': records}
    if extra:
        manifest.update(extra)
    (Path(run_dir) / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
