"""Predict aligned RGB/terrain pairs with explicit real-DEM or FHM routing."""
import argparse
import warnings
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from utils.inference import (add_common_arguments, load_inference_model, is_dual_model, missing_route,
                             evaluation_options, create_run_directory, write_manifest)
from utils.modalities import index_images, list_rgb_images, load_inference_pair, read_mask, require_same_size
from utils.engine import predict_probabilities, probabilities_to_masks
from utils.stream_metrics import StreamSegMetrics


def blend(image, selection, color, alpha):
    image[selection] = image[selection] * (1 - alpha) + np.asarray(color) * alpha


def main(opts):
    if not 0 <= opts.alpha <= 1:
        raise ValueError('--alpha must be in [0,1]')
    paths = list_rgb_images(opts.input)
    model, config, device, report = load_inference_model(opts)
    dual = is_dual_model(config)
    mode = opts.dem_mode if dual else 'missing'
    if not dual and opts.compare_dem:
        raise ValueError('--compare_dem requires a dual-stream model')
    dems = index_images(opts.dem_dir) if mode != 'missing' else {}
    targets = index_images(opts.gt_dir) if opts.gt_dir else {}
    run_dir = create_run_directory(opts.output_dir, 'predict', mode if dual else 'rgb')
    for name, enabled in [('mask', opts.save_mask), ('overlay', opts.save_overlay),
                          ('analysis', opts.save_analysis and bool(opts.gt_dir)), ('probability', opts.save_probs)]:
        if enabled:
            (run_dir / name).mkdir()
    eval_opts = evaluation_options(config)
    metrics = StreamSegMetrics(2)
    evaluated = 0
    records = []
    with torch.inference_mode():
        for path in tqdm(paths, desc='Predict'):
            original, rgb, dem, present, dem_path = load_inference_pair(path, dems, mode, config['resize'])
            rgb, dem = rgb.unsqueeze(0).to(device), dem.unsqueeze(0).to(device)
            has_dem = torch.tensor([present], dtype=torch.bool, device=device)
            logits, probs = predict_probabilities(eval_opts, model, rgb, dem, has_dem)
            # Match validation's argmax grid; resize only the exported mask.
            grid_mask = probabilities_to_masks(probs)[0]
            mask = np.array(Image.fromarray(grid_mask).resize(original.size, Image.Resampling.NEAREST))
            row = {'rgb': str(path.resolve()), 'dem': str(dem_path.resolve()) if dem_path else None,
                   'has_dem': present, 'route': 'real_dem' if present and dual else missing_route(config),
                   'original_size': list(original.size), 'network_size': list(rgb.shape[-2:])}
            if opts.compare_dem and present:
                fake_logits, fake_probs = predict_probabilities(eval_opts, model, rgb, dem, has_dem, force_missing=True)
                fake_mask = probabilities_to_masks(fake_probs)[0]
                row['real_vs_missing'] = {
                    'logit_max_abs': float((logits - fake_logits).abs().max()),
                    'probability_mean_abs': float((probs[:, 1] - fake_probs[:, 1]).abs().mean()),
                    'argmax_changed_fraction': float((logits.argmax(1) != fake_logits.argmax(1)).float().mean()),
                    'mask_changed_fraction': float(np.mean(grid_mask != fake_mask))}
            if opts.save_mask:
                Image.fromarray(mask).save(run_dir / 'mask' / (path.stem + '.png'))
            if opts.save_overlay:
                overlay = np.array(original, dtype=np.float32)
                blend(overlay, mask == 1, (255, 255, 0), opts.alpha)
                Image.fromarray(overlay.astype(np.uint8)).save(run_dir / 'overlay' / (path.stem + '.png'))
            if opts.save_probs:
                np.save(run_dir / 'probability' / (path.stem + '.npy'), probs[0, 1].cpu().numpy())
            gt_path = targets.get(path.stem.casefold())
            if gt_path is not None:
                gt_image = read_mask(gt_path, config['mask_encoding'])
                require_same_size(original, gt_image, f'GT {gt_path}')
                # Metrics on the validation grid; exported error visualization at native size.
                grid_gt = np.array(gt_image.resize((rgb.shape[-1], rgb.shape[-2]), Image.Resampling.NEAREST))
                metrics.update(grid_gt[None], grid_mask[None])
                evaluated += 1
                if opts.save_analysis:
                    gt = np.array(gt_image)
                    analysis = np.array(original, dtype=np.float32)
                    blend(analysis, (mask == 1) & (gt == 1), (0, 255, 0), opts.alpha)
                    blend(analysis, (mask == 1) & (gt == 0), (0, 0, 255), opts.alpha)
                    blend(analysis, (mask == 0) & (gt == 1), (255, 0, 0), opts.alpha)
                    Image.fromarray(analysis.astype(np.uint8)).save(run_dir / 'analysis' / (path.stem + '.png'))
            elif opts.gt_dir:
                warnings.warn(f'No GT for {path.name}; excluded from metrics')
            records.append(row)
            print(f"[Route] {path.name}: {row['route']}")
    extra = {'evaluated_images': evaluated, 'metrics_grid': 'network input',
             'probability_grid': 'network input', 'exported_mask_grid': 'original RGB'}
    if evaluated:
        score = metrics.get_results()
        extra['metrics'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in score.items()}
        print(f"[Metrics] mIoU={score['Mean IoU']:.6f}, landslide IoU={score['Class IoU'][1]:.6f}")
    write_manifest(run_dir, opts, config, report, records, extra)
    print(f'[Done] {run_dir.resolve()}')
    return run_dir


def get_argparser():
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--gt_dir', default=None)
    parser.add_argument('--alpha', type=float, default=0.5, help='Visualization opacity only')
    for name in ('save_mask', 'save_overlay', 'save_analysis'):
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_probs', action='store_true')
    parser.add_argument('--compare_dem', action='store_true', help='Record real vs missing DEM logit/probability/mask differences')
    return parser


if __name__ == '__main__':
    main(get_argparser().parse_args())
