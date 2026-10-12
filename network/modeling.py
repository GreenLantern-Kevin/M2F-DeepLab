"""DeepLabV3+ factories and explicit RGB/DEM routing."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import mobilenetv2, resnet, xception
from ._deeplab import DeepLabHeadV3Plus, DeepLabV3Plus, MEDeepLabHeadV3Plus
from .utils import IntermediateLayerGetter, _MEDeepLabModelWrapper


DUAL_MODEL_NAMES = (
    "me_deeplabv3plus_resnet50",
    "m2f_deeplabv3plus_xception16",
    "m2f_deeplabv3plus_resnet50",
    "m2f_deeplabv3plus_xception16_b",
    "m2f_deeplabv3plus_xception16_c",
    "m2f_deeplabv3plus_xception16_d",
)
ZERO_FALLBACK_MODEL_NAMES = ("m2f_deeplabv3plus_xception16_b",)
SINGLE_MODEL_NAMES = (
    "deeplabv3plus_resnet50", "deeplabv3plus_resnet101", "deeplabv3plus_mobilenet",
    "deeplabv3plus_xception8", "deeplabv3plus_xception16",
    "single_deeplabv3plus_resnet50", "single_deeplabv3plus_xception16",
)
MODEL_NAMES = SINGLE_MODEL_NAMES + DUAL_MODEL_NAMES
__all__ = list(MODEL_NAMES) + ["MODEL_NAMES", "SINGLE_MODEL_NAMES", "DUAL_MODEL_NAMES", "ZERO_FALLBACK_MODEL_NAMES"]


def _aspp_rates(output_stride, small_tiles=False):
    if output_stride not in (8, 16):
        raise ValueError("output_stride must be 8 or 16")
    rates = (4, 8, 12) if small_tiles else (12, 24, 36)
    return rates if output_stride == 8 else tuple(rate // 2 for rate in rates)


def _resnet_backbone(name, output_stride, pretrained):
    _aspp_rates(output_stride)
    dilation = [False, True, True] if output_stride == 8 else [False, False, True]
    net = getattr(resnet, name)(pretrained=pretrained, replace_stride_with_dilation=dilation)
    layers = {"layer4": "out", "layer1": "low_level"}
    return IntermediateLayerGetter(net, return_layers=layers)


def _load_xception_pretrained(*backbones):
    """Load project-local ImageNet weights only when explicitly requested."""
    path = Path(__file__).resolve().parents[1] / "weight" / "xception_pytorch_imagenet.pth"
    if not path.is_file():
        raise FileNotFoundError(
            f"Requested Xception pretrained weights are missing: {path}. "
            "Use pretrained_backbone=False for random initialization or checkpoint loading."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    for backbone in backbones:
        matched = {}
        for name, target in backbone.state_dict().items():
            for key, source in state.items():
                if torch.is_tensor(source) and name.endswith(key) and target.shape == source.shape:
                    matched[name] = source
                    break
        if not matched:
            raise RuntimeError(f"No Xception parameters matched local weights: {path}")
        backbone.load_state_dict(matched, strict=False)
        print(f"[Pretrained] Xception: loaded {len(matched)} tensors from {path}")


def _segm_resnet(backbone_name, num_classes, output_stride, pretrained_backbone):
    backbone = _resnet_backbone(backbone_name, output_stride, pretrained_backbone)
    head = DeepLabHeadV3Plus(2048, 256, num_classes, _aspp_rates(output_stride))
    return DeepLabV3Plus(backbone, head)


def _segm_xception(backbone_name, num_classes, output_stride, pretrained_backbone):
    variant = "xception16" if backbone_name == "xception" else backbone_name
    backbone = xception.XceptionBackbone(output_stride, False, variant)
    if pretrained_backbone:
        _load_xception_pretrained(backbone)
    head = DeepLabHeadV3Plus(2048, 256, num_classes, _aspp_rates(output_stride))
    return DeepLabV3Plus(backbone, head)


def _segm_mobilenet(backbone_name, num_classes, output_stride, pretrained_backbone):
    rates = _aspp_rates(output_stride)
    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None
    backbone = IntermediateLayerGetter(
        backbone, {"high_level_features": "out", "low_level_features": "low_level"}
    )
    return DeepLabV3Plus(backbone, DeepLabHeadV3Plus(320, 24, num_classes, rates))


def _load_model(backbone, num_classes, output_stride, pretrained_backbone):
    if backbone == "mobilenetv2":
        factory = _segm_mobilenet
    elif backbone.startswith("resnet"):
        factory = _segm_resnet
    elif backbone in ("xception", "xception16", "xception8"):
        factory = _segm_xception
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return factory(backbone, num_classes, output_stride, pretrained_backbone)


def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model("resnet50", num_classes, output_stride, pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model("resnet101", num_classes, output_stride, pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model("mobilenetv2", num_classes, output_stride, pretrained_backbone)


def deeplabv3plus_xception8(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model("xception8", num_classes, output_stride, pretrained_backbone)


def deeplabv3plus_xception16(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model("xception16", num_classes, output_stride, pretrained_backbone)


class FeatureHallucinationModule(nn.Module):
    """Map RGB features into the DEM feature space."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.map(x)


class ConcatFusion(nn.Module):
    """Models B/C: concatenate high-level features, then project back to RGB width."""

    def __init__(self, rgb_ch, topo_ch, out_ch):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(rgb_ch + topo_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_rgb, feat_topo):
        return self.project(torch.cat([feat_rgb, feat_topo], dim=1))


class ModalityElasticFusion(nn.Module):
    """RGB-guided spatial/channel attention and residual fusion.

    A small nonzero final BN scale permits first-step DEM task gradients.
    Loading a checkpoint restores its saved scale.
    """

    def __init__(self, rgb_ch, topo_ch, out_ch, fusion_init_scale=0.1):
        super().__init__()
        if rgb_ch != out_ch:
            raise ValueError("Residual fusion requires rgb_ch == out_ch")
        if not math.isfinite(fusion_init_scale) or fusion_init_scale < 0:
            raise ValueError("fusion_init_scale must be finite and nonnegative")
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(rgb_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(rgb_ch + topo_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        nn.init.constant_(self.fuse[-1].weight, fusion_init_scale)
        nn.init.constant_(self.fuse[-1].bias, 0)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(1, out_ch // 16), 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_ch // 16), out_ch, 1, bias=False), nn.Sigmoid(),
        )

    def forward(self, feat_rgb, feat_topo):
        topo = feat_topo * self.spatial_gate(feat_rgb)
        fused = self.fuse(torch.cat([feat_rgb, topo], dim=1))
        return feat_rgb + fused * self.channel_gate(fused)


class MEDeepLabDualStreamBackbone(nn.Module):
    """Route each sample to real DEM features or its available substitute."""

    def __init__(self, rgb_backbone, dem_backbone, rgb_ch=2048, dem_ch=512,
                 fusion_init_scale=0.1, *, use_fhm=True, fusion="mef"):
        super().__init__()
        if fusion not in ("concat", "mef"):
            raise ValueError("fusion must be 'concat' or 'mef'")
        self.rgb_backbone = rgb_backbone
        self.dem_backbone = dem_backbone
        self.dem_ch = dem_ch
        self.fhm = FeatureHallucinationModule(rgb_ch, dem_ch) if use_fhm else None
        self.mef = ModalityElasticFusion(rgb_ch, dem_ch, rgb_ch, fusion_init_scale) if fusion == "mef" else None
        self.concat = ConcatFusion(rgb_ch, dem_ch, rgb_ch) if fusion == "concat" else None

    @staticmethod
    def _mask(value, name, batch_size, device):
        if not torch.is_tensor(value) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be a torch.bool tensor of shape [B]")
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [{batch_size}], got {tuple(value.shape)}")
        return value.to(device=device)

    def forward(self, x_rgb, x_dem=None, has_dem=None, dem_drop_mask=None):
        if (not torch.is_tensor(x_rgb) or x_rgb.ndim != 4 or x_rgb.shape[1] != 3
                or x_rgb.shape[0] == 0 or not x_rgb.is_floating_point()):
            raise ValueError("x_rgb must be a nonempty floating [B, 3, H, W] tensor")
        batch_size = x_rgb.shape[0]
        if x_dem is not None:
            if has_dem is None:
                raise ValueError("Passing x_dem requires explicit has_dem; pixel values do not indicate availability")
            if not torch.is_tensor(x_dem) or x_dem.shape != x_rgb.shape:
                raise ValueError("x_dem must have the same [B, 3, H, W] shape as x_rgb")
            if x_dem.device != x_rgb.device or x_dem.dtype != x_rgb.dtype:
                raise ValueError("x_dem and x_rgb must have the same device and floating dtype")
        if has_dem is None:
            has_dem = torch.zeros(batch_size, dtype=torch.bool, device=x_rgb.device)
        has_dem = self._mask(has_dem, "has_dem", batch_size, x_rgb.device)
        if x_dem is None and has_dem.any():
            raise ValueError("has_dem cannot contain True when x_dem is None")
        if dem_drop_mask is None:
            dem_drop_mask = torch.zeros_like(has_dem)
        dem_drop_mask = self._mask(dem_drop_mask, "dem_drop_mask", batch_size, x_rgb.device)
        if (dem_drop_mask & ~has_dem).any():
            raise ValueError("dem_drop_mask may only drop samples with has_dem=True")
        used_dem = has_dem & ~dem_drop_mask

        rgb_feats = self.rgb_backbone(x_rgb)
        rgb_high = rgb_feats["out"]
        # Consistency training updates FHM without altering its RGB input branch.
        feat_fake = self.fhm(rgb_high.detach()) if self.fhm is not None else None
        empty_topo = rgb_high.new_zeros((batch_size, self.dem_ch, *rgb_high.shape[2:]))
        feat_real = empty_topo
        if has_dem.any():
            indices = has_dem.nonzero(as_tuple=True)[0]
            # Keep teachers for dropped real DEMs; never feed missing placeholders to BN.
            try:
                real = self.dem_backbone(x_dem.index_select(0, indices))["out"]
            except ValueError as error:
                if "Expected more than 1 value per channel" not in str(error):
                    raise
                raise ValueError(
                    "DEM BatchNorm received only one value per channel in the valid-DEM subbatch. "
                    "Use larger inputs, more real DEM samples per device, or explicitly freeze "
                    "BatchNorm (--freeze_bn). Missing placeholders are never added to this subbatch."
                ) from error
            if real.shape[2:] != rgb_high.shape[2:]:
                real = F.interpolate(real, size=rgb_high.shape[2:], mode="bilinear", align_corners=False)
            feat_real = feat_real.index_copy(0, indices, real)

        substitute = feat_fake if feat_fake is not None else empty_topo
        topo = torch.where(used_dem[:, None, None, None], feat_real, substitute)
        fused = self.mef(rgb_high, topo) if self.mef is not None else self.concat(rgb_high, topo)
        return {
            "out": fused, "low_level": rgb_feats["low_level"],
            "feat_fake": feat_fake, "feat_real": feat_real,
            "has_dem": has_dem, "used_dem": used_dem,
            "used_fhm": (~used_dem) & (self.fhm is not None),
        }


def me_deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True,
                             fusion_init_scale=0.1):
    """ResNet50 RGB / ResNet18 DEM variant."""
    from torchvision.models.resnet import resnet18
    rgb = _resnet_backbone("resnet50", output_stride, pretrained_backbone)
    dem = IntermediateLayerGetter(resnet18(pretrained=pretrained_backbone), {"layer4": "out"})
    backbone = MEDeepLabDualStreamBackbone(rgb, dem, fusion_init_scale=fusion_init_scale)
    head = MEDeepLabHeadV3Plus(2048, 256, num_classes, _aspp_rates(output_stride))
    return _MEDeepLabModelWrapper(backbone, head)


def m2f_deeplabv3plus_xception16(num_classes=21, output_stride=8, pretrained_backbone=True,
                               fusion_init_scale=0.1):
    rgb = xception.XceptionBackbone(output_stride, False, "xception16")
    dem = xception.XceptionBackbone(output_stride, False, "xception16")
    if pretrained_backbone:
        _load_xception_pretrained(rgb, dem)
    backbone = MEDeepLabDualStreamBackbone(rgb, dem, 2048, 2048, fusion_init_scale)
    head = MEDeepLabHeadV3Plus(2048, 256, num_classes, _aspp_rates(output_stride, True))
    return _MEDeepLabModelWrapper(backbone, head)


def m2f_deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True,
                             fusion_init_scale=0.1):
    rgb = _resnet_backbone("resnet50", output_stride, pretrained_backbone)
    dem = _resnet_backbone("resnet50", output_stride, pretrained_backbone)
    backbone = MEDeepLabDualStreamBackbone(rgb, dem, 2048, 2048, fusion_init_scale)
    head = MEDeepLabHeadV3Plus(2048, 256, num_classes, _aspp_rates(output_stride, True))
    return _MEDeepLabModelWrapper(backbone, head)


def _xception_ablation(num_classes, output_stride, pretrained_backbone, fusion_init_scale,
                       *, use_fhm, fusion):
    """Models B/C/D share the full model's encoders/ASPP rates; none contains BEM."""
    rgb = xception.XceptionBackbone(output_stride, False, "xception16")
    dem = xception.XceptionBackbone(output_stride, False, "xception16")
    if pretrained_backbone:
        _load_xception_pretrained(rgb, dem)
    backbone = MEDeepLabDualStreamBackbone(
        rgb, dem, 2048, 2048, fusion_init_scale, use_fhm=use_fhm, fusion=fusion)
    head = DeepLabHeadV3Plus(2048, 256, num_classes, _aspp_rates(output_stride, True))
    return _MEDeepLabModelWrapper(backbone, head)


def m2f_deeplabv3plus_xception16_b(num_classes=2, output_stride=16, pretrained_backbone=True,
                                 fusion_init_scale=0.1):
    """Model B: real DEM or zero features, basic concatenation, no FHM/MEF/BEM."""
    return _xception_ablation(num_classes, output_stride, pretrained_backbone, fusion_init_scale,
                              use_fhm=False, fusion="concat")


def m2f_deeplabv3plus_xception16_c(num_classes=2, output_stride=16, pretrained_backbone=True,
                                 fusion_init_scale=0.1):
    """Model C: real DEM or FHM, basic concatenation, no MEF/BEM."""
    return _xception_ablation(num_classes, output_stride, pretrained_backbone, fusion_init_scale,
                              use_fhm=True, fusion="concat")


def m2f_deeplabv3plus_xception16_d(num_classes=2, output_stride=16, pretrained_backbone=True,
                                 fusion_init_scale=0.1):
    """Model D: real DEM or FHM, MEF, ordinary decoder without BEM."""
    return _xception_ablation(num_classes, output_stride, pretrained_backbone, fusion_init_scale,
                              use_fhm=True, fusion="mef")


def single_deeplabv3plus_resnet50(num_classes=2, output_stride=8, pretrained_backbone=True):
    backbone = _resnet_backbone("resnet50", output_stride, pretrained_backbone)
    head = DeepLabHeadV3Plus(2048, 256, num_classes, _aspp_rates(output_stride, True))
    return DeepLabV3Plus(backbone, head)


class _XceptionToDictWrapper(nn.Module):
    """Adapt an Xception encoder to the decoder feature dictionary."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        low, high = self.net(x)
        return {"low_level": low, "out": high}


def single_deeplabv3plus_xception16(num_classes=2, output_stride=8, pretrained_backbone=False):
    rates = _aspp_rates(output_stride, True)
    raw = xception.xception16(pretrained=False, downsample_factor=output_stride)
    if pretrained_backbone:
        _load_xception_pretrained(raw)
    backbone = _XceptionToDictWrapper(raw)
    head = DeepLabHeadV3Plus(2048, 256, num_classes, rates)
    return DeepLabV3Plus(backbone, head)
