from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset

from .modalities import index_images, read_rgb, read_terrain, read_mask, read_edge, require_same_size


class WHULandslideDataset(Dataset):
    """Aligned RGB/terrain segmentation dataset with explicit DEM availability."""

    def __init__(self, root_dir, split='train', transform=None, mask_encoding='indices'):
        super().__init__()
        self.root_dir, self.split, self.transform = root_dir, split, transform
        self.mask_encoding = mask_encoding
        base = Path(root_dir) / 'VOC2012'
        self.img_dir, self.dem_dir = base / 'JPEGImages', base / 'DEMImages'
        self.mask_dir, self.edge_dir = base / 'SegmentationClass', base / 'EdgeClass'
        split_file = base / 'ImageSets' / 'Segmentation' / f'{split}.txt'
        self.file_names = [line.strip() for line in split_file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        self.rgb_index, self.dem_index = index_images(self.img_dir), index_images(self.dem_dir)
        self.mask_index, self.edge_index = index_images(self.mask_dir), index_images(self.edge_dir)
        for name in self.file_names:
            key = Path(name).stem.casefold()
            for kind, lookup in (('RGB', self.rgb_index), ('mask', self.mask_index), ('edge', self.edge_index)):
                if key not in lookup:
                    raise FileNotFoundError(f'{split} sample {name}: missing {kind} in {base}')
        present = sum(Path(name).stem.casefold() in self.dem_index for name in self.file_names)
        self.dem_coverage = {'total': len(self.file_names), 'present': present,
                             'missing': len(self.file_names) - present,
                             'fraction': present / len(self.file_names) if self.file_names else 0.0}

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        key = Path(self.file_names[index]).stem.casefold()
        img = read_rgb(self.rgb_index[key])
        mask = read_mask(self.mask_index[key], self.mask_encoding)
        edge = read_edge(self.edge_index[key])
        require_same_size(img, mask, f'Mask {self.mask_index[key]}')
        require_same_size(img, edge, f'Edge {self.edge_index[key]}')
        has_dem = key in self.dem_index
        dem = read_terrain(self.dem_index[key], img.size) if has_dem else Image.new('RGB', img.size)
        if self.transform is not None:
            img, dem, mask, edge = self.transform(img, dem, mask, edge)
        # Route with this flag, never with pixel values after normalization.
        return img, dem, mask, edge, torch.tensor(has_dem, dtype=torch.bool)
