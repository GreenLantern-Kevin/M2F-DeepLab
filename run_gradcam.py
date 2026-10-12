"""One Grad-CAM implementation for real DEM and missing DEM comparisons."""
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from utils.inference import (add_common_arguments, load_inference_model, forward_outputs,
                             is_dual_model, missing_route, create_run_directory, write_manifest)
from utils.modalities import index_images, list_rgb_images, load_inference_pair


class SemanticSegmentationTarget:
    def __init__(self, category_index=1):
        self.category_index = category_index

    def __call__(self, model_output):
        return model_output[self.category_index].sum()


class M2FWrapper(torch.nn.Module):
    def __init__(self, model, config, dem_tensor, has_dem):
        super().__init__()
        self.model, self.config = model, config
        self.dem_tensor, self.has_dem = dem_tensor, has_dem

    def forward(self, rgb_tensor):
        return forward_outputs(self.model, self.config, rgb_tensor, self.dem_tensor, self.has_dem)['seg']


def main(opts):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as error:
        raise RuntimeError('Grad-CAM requires the optional grad-cam package on your analysis machine.') from error
    model, config, device, report = load_inference_model(opts)
    modules = dict(model.named_modules())
    if opts.target_layer not in modules:
        raise ValueError(f'Unknown target layer: {opts.target_layer}')
    mode = opts.dem_mode if is_dual_model(config) else 'missing'
    dems = index_images(opts.dem_dir) if mode != 'missing' else {}
    run_dir = create_run_directory(opts.output_dir, 'gradcam', mode)
    records = []
    print(f'[CAM] layer={opts.target_layer}; class=1; full-image logit sum; single forward')
    for path in tqdm(list_rgb_images(opts.input), desc='Grad-CAM'):
        original, rgb, dem, present, dem_path = load_inference_pair(path, dems, mode, config['resize'])
        rgb, dem = rgb.unsqueeze(0).to(device), dem.unsqueeze(0).to(device)
        has_dem = torch.tensor([present], device=device, dtype=torch.bool)
        wrapper = M2FWrapper(model, config, dem, has_dem)
        # A context manager removes hooks after each sample.
        with GradCAM(model=wrapper, target_layers=[modules[opts.target_layer]]) as cam:
            heatmap = cam(input_tensor=rgb, targets=[SemanticSegmentationTarget()])[0]
        np.save(run_dir / (path.stem + '_cam.npy'), heatmap)
        background = original.resize((rgb.shape[-1], rgb.shape[-2]), Image.Resampling.BILINEAR)
        overlay = show_cam_on_image(np.asarray(background, dtype=np.float32) / 255, heatmap, use_rgb=True)
        Image.fromarray(overlay).save(run_dir / (path.stem + '.png'))
        records.append({'rgb': str(path.resolve()), 'dem': str(dem_path.resolve()) if dem_path else None,
                        'has_dem': present, 'route': 'real_dem' if present else missing_route(config)})
    write_manifest(run_dir, opts, config, report, records,
                   {'cam': {'layer': opts.target_layer, 'class': 1, 'target': 'full-image logit sum',
                            'normalization': 'per-image Grad-CAM; visual similarity is not modality importance'}})
    print(f'[Done] {run_dir.resolve()}')


def get_argparser(default_dem_mode='required'):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__), default_dem_mode)
    parser.add_argument('--target_layer', default='classifier.classifier.0', help='Explicit executed main decoder layer')
    return parser


if __name__ == '__main__':
    main(get_argparser().parse_args())
