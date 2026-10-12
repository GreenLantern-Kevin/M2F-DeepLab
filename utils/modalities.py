"""Shared RGB/terrain loading; terrain channels are elevation, slope, aspect.

Inputs must already be spatially registered. Equal dimensions cannot verify
geographic registration. Single-band elevation is not preprocessed terrain RGB.
"""
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def index_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f'Ambiguous image stem {path.stem!r}: {result[key]} and {path}')
            result[key] = path
    return result


def list_rgb_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported RGB image format: {path}')
        return [path]
    paths = list(index_images(path).values())
    if not paths:
        raise FileNotFoundError(f'No RGB images found: {path}')
    return paths


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def require_same_size(reference, other, description):
    if reference.size != other.size:
        raise ValueError(f'{description}: size {other.size} differs from RGB {reference.size}. '
                         'Use registered inputs; resizing cannot establish registration.')


def read_terrain(path, expected_size=None):
    path = Path(path)
    # Pillow can downcast 16-bit RGB PNG; inspect IHDR before decoding.
    with path.open('rb') as stream:
        header = stream.read(26)
    if header.startswith(b'\x89PNG\r\n\x1a\n') and header[24] != 8:
        raise ValueError(f'DEM must be preprocessed 8-bit terrain RGB: {path}')
    with Image.open(path) as image:
        if image.format == 'TIFF':
            bits = image.tag_v2.get(258, (8,))
            bits = (bits,) if isinstance(bits, int) else bits
            if any(bit != 8 for bit in bits):
                raise ValueError(f'DEM TIFF must contain three 8-bit terrain channels: {path}')
        array = np.asarray(image)
        if image.mode != 'RGB' or array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f'DEM must be three-channel uint8 RGB (elevation/slope/aspect); '
                             f'got mode={image.mode}, shape={array.shape}, dtype={array.dtype}: {path}. '
                             'Raw single-channel DEM is not automatically replicated.')
        if expected_size is not None and image.size != tuple(expected_size):
            raise ValueError(f'DEM size {image.size} differs from RGB {expected_size}: {path}; align upstream.')
        return image.copy()


def read_mask(path, encoding='indices'):
    """Preserve palette indices; 255 is ignore unless explicitly binary255."""
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f'Mask must be single-channel/indexed: {path}')
    if encoding not in ('indices', 'binary255'):
        raise ValueError(f'Unknown mask encoding: {encoding}')
    allowed = (0, 1, 255) if encoding == 'indices' else (0, 255)
    if not np.isin(array, allowed).all():
        raise ValueError(f'Mask values must be {allowed} for {encoding}: {path}; found {np.unique(array)}')
    if encoding == 'binary255':
        array = (array == 255).astype(np.uint8)
    return Image.fromarray(array.astype(np.uint8))


def read_edge(path):
    """Disk white 255 is foreground; later augmentation reserves 255 for ignore."""
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2 or not np.isin(array, (0, 1, 255)).all():
        raise ValueError(f'Edge must contain 0/1 or 0/255 values: {path}')
    return Image.fromarray((array > 0).astype(np.uint8))


def image_to_tensor(image):
    array = np.array(image, dtype=np.uint8, copy=True)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f'Expected three channels, got {array.shape}')
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)


def normalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return (tensor - tensor.new_tensor(mean).view(3, 1, 1)) / tensor.new_tensor(std).view(3, 1, 1)


def load_inference_pair(rgb_path, dem_index, dem_mode='auto', resize=None):
    """Return original RGB, processed RGB/DEM, explicit availability, DEM path.

    Optional square resize is joint and follows original-size validation.
    Missing DEM stays zero after preprocessing; routing uses the boolean flag.
    """
    if dem_mode not in ('auto', 'required', 'missing'):
        raise ValueError(f'Unknown dem_mode: {dem_mode}')
    original = read_rgb(rgb_path)
    dem_path = dem_index.get(Path(rgb_path).stem.casefold()) if dem_mode != 'missing' else None
    has_dem = dem_path is not None
    if not has_dem and dem_mode == 'required':
        raise FileNotFoundError(f'Required DEM missing for RGB: {rgb_path}')
    if not has_dem and dem_mode == 'auto':
        warnings.warn(f'DEM missing for {Path(rgb_path).name}; using the model\'s missing-DEM route.', RuntimeWarning)
    dem = read_terrain(dem_path, original.size) if has_dem else None
    rgb = original
    if resize is not None:
        size = (int(resize), int(resize))
        if size[0] < 1:
            raise ValueError('resize must be positive, or None for original size')
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
        if has_dem:
            dem = dem.resize(size, Image.Resampling.BILINEAR)
    rgb_tensor = normalize_tensor(image_to_tensor(rgb))
    dem_tensor = normalize_tensor(image_to_tensor(dem)) if has_dem else torch.zeros_like(rgb_tensor)
    return original, rgb_tensor, dem_tensor, has_dem, dem_path
