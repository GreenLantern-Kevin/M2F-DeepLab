"""Joint spatial transforms; color jitter is applied only to RGB."""
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from .modalities import IMAGENET_MEAN, IMAGENET_STD, image_to_tensor, normalize_tensor


class M2FCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, dem, mask, edge):
        for transform in self.transforms:
            img, dem, mask, edge = transform(img, dem, mask, edge)
        return img, dem, mask, edge


class M2FValResize:
    def __init__(self, size=(320, 320)):
        self.size = size

    def __call__(self, img, dem, mask, edge):
        return (F.resize(img, self.size, InterpolationMode.BILINEAR),
                F.resize(dem, self.size, InterpolationMode.BILINEAR),
                F.resize(mask, self.size, InterpolationMode.NEAREST),
                F.resize(edge, self.size, InterpolationMode.NEAREST))


class M2FRandomScale:
    def __init__(self, scale_range=(0.8, 1.2), base_size=(320, 320)):
        self.scale_range, self.base_size = scale_range, base_size

    def __call__(self, img, dem, mask, edge):
        # Retain the original base resize followed by scale jitter.
        images = M2FValResize(self.base_size)(img, dem, mask, edge)
        scale = random.uniform(*self.scale_range)
        size = tuple(max(1, int(length * scale)) for length in self.base_size)
        return M2FValResize(size)(*images)


class M2FRandomCrop:
    def __init__(self, size, pad_if_needed=True):
        self.size, self.pad_if_needed = (int(size), int(size)), pad_if_needed

    def __call__(self, img, dem, mask, edge):
        pad_w, pad_h = max(0, self.size[1] - img.width), max(0, self.size[0] - img.height)
        if self.pad_if_needed and (pad_w or pad_h):
            padding = (0, 0, pad_w, pad_h)
            img, dem = F.pad(img, padding, fill=0), F.pad(dem, padding, fill=0)
            mask, edge = F.pad(mask, padding, fill=255), F.pad(edge, padding, fill=255)
        crop = T.RandomCrop.get_params(img, output_size=self.size)
        return tuple(F.crop(image, *crop) for image in (img, dem, mask, edge))


class M2FRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, dem, mask, edge):
        images = (img, dem, mask, edge)
        return tuple(F.hflip(image) for image in images) if random.random() < self.p else images


class M2FToTensor:
    def __call__(self, img, dem, mask, edge):
        labels, edges = np.array(mask, dtype=np.int64), np.array(edge, dtype=np.float32)
        if not np.isin(labels, (0, 1, 255)).all() or not np.isin(edges, (0, 1, 255)).all():
            raise ValueError('Targets must contain 0/1 and padding/ignore 255')
        return image_to_tensor(img), image_to_tensor(dem), torch.from_numpy(labels), torch.from_numpy(edges).unsqueeze(0)


class M2FNormalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean, self.std = mean, std

    def __call__(self, img, dem, mask, edge):
        return (normalize_tensor(img, self.mean, self.std), normalize_tensor(dem, self.mean, self.std), mask, edge)


class M2FColorJitter:
    def __init__(self, brightness=0.32, contrast=0.5, saturation=0.5, hue=0.1):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, dem, mask, edge):
        return self.jitter(img), dem, mask, edge
