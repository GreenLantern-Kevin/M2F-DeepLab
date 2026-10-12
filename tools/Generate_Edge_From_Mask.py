"""Generate morphological boundaries from class-index masks, preserving existing edges."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def generate_edges_from_masks(mask_dir, output_edge_dir, mask_encoding='indices'):
    output = Path(output_edge_dir)
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(mask_dir).glob('*.png')):
        destination = output / path.name
        if destination.exists():
            continue
        with Image.open(path) as image:
            mask = np.array(image)
        allowed = (0, 1, 255) if mask_encoding == 'indices' else (0, 255)
        if mask.ndim != 2 or not np.isin(mask, allowed).all():
            raise ValueError(f'Invalid mask: {path}')
        foreground = 1 if mask_encoding == 'indices' else 255
        binary = (mask == foreground).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        if mask_encoding == 'indices':
            edge[cv2.dilate((mask == 255).astype(np.uint8), kernel) > 0] = 0
        Image.fromarray(edge).save(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--output_edge_dir', required=True)
    parser.add_argument('--mask_encoding', choices=('indices', 'binary255'), default='indices')
    opts = parser.parse_args()
    generate_edges_from_masks(opts.mask_dir, opts.output_edge_dir, opts.mask_encoding)
