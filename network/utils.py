"""Feature extraction and segmentation output wrappers."""
from collections import OrderedDict

from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        size = x.shape[-2:]
        result = self.classifier(self.backbone(x))
        if isinstance(result, dict):
            return {
                key: F.interpolate(value, size=size, mode="bilinear", align_corners=False)
                if value is not None else None
                for key, value in result.items()
            }
        return F.interpolate(result, size=size, mode="bilinear", align_corners=False)


class IntermediateLayerGetter(nn.ModuleDict):
    """Collect named direct children while preserving their parameter names."""

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(dict(model.named_children())):
            raise ValueError("return_layers are not present in model")
        requested = dict(return_layers)
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            requested.pop(name, None)
            if not requested:
                break
        super().__init__(layers)
        self.return_layers = dict(return_layers)

    def forward(self, x):
        output = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                output[self.return_layers[name]] = x
        return output


class _MEDeepLabModelWrapper(nn.Module):
    """Return full-resolution logits and explicit DEM routing metadata."""

    def __init__(self, dual_backbone, classifier):
        super().__init__()
        self.backbone = dual_backbone
        self.classifier = classifier

    def forward(self, x_rgb, x_dem=None, has_dem=None, dem_drop_mask=None):
        features = self.backbone(x_rgb, x_dem, has_dem=has_dem, dem_drop_mask=dem_drop_mask)
        head = self.classifier(features)
        if not isinstance(head, dict):
            head = {"seg": head}
        result = {}
        for key in ("seg", "edge"):
            value = head.get(key)
            result[key] = (
                F.interpolate(value, size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)
                if value is not None else None
            )
        result.update({key: features[key] for key in ("feat_fake", "feat_real", "has_dem", "used_dem", "used_fhm")})
        return result
