"""Segmentation and boundary losses; 255 always denotes ignored pixels."""

import torch
from torch import nn
from torch.nn import functional as F


def _valid_pixels(predict, target, ignore_index):
    if predict.ndim != 4 or target.shape != predict.shape[:1] + predict.shape[2:]:
        raise ValueError("Expected logits [B,C,H,W] and class indices [B,H,W].")
    valid = target != ignore_index
    logits = predict.permute(0, 2, 3, 1)[valid]
    labels = target[valid].long()
    if labels.numel() and ((labels < 0).any() or (labels >= predict.shape[1]).any()):
        raise ValueError("Mask contains a class index outside the model's classes.")
    return logits, labels


def cross_entropy_valid(predict, target, ignore_index=255):
    logits, labels = _valid_pixels(predict, target, ignore_index)
    return F.cross_entropy(logits, labels) if labels.numel() else predict.sum() * 0.0


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255, reduction="mean", **kwargs):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("Unsupported focal loss reduction.")
        self.alpha, self.gamma = alpha, gamma
        self.ignore_index, self.reduction = ignore_index, reduction

    def forward(self, predict, target):
        logits, labels = _valid_pixels(predict, target, self.ignore_index)
        if not labels.numel():
            return predict.sum() * 0.0
        ce = F.cross_entropy(logits, labels, reduction="none")
        loss = self.alpha * (1.0 - torch.exp(-ce)).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum() if self.reduction == "sum" else loss


class DiceLoss(nn.Module):
    """Mean soft Dice over all classes."""

    def __init__(self, ignore_index=255, smooth=1e-5, **kwargs):
        super().__init__()
        self.ignore_index, self.smooth = ignore_index, smooth

    def forward(self, predict, target):
        logits, labels = _valid_pixels(predict, target, self.ignore_index)
        if not labels.numel():
            return predict.sum() * 0.0
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(labels, num_classes=predict.shape[1]).to(probabilities.dtype)
        intersection = (probabilities * one_hot).sum(dim=0)
        denominator = probabilities.sum(dim=0) + one_hot.sum(dim=0)
        return 1.0 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class LandslideDiceLoss(DiceLoss):
    """Soft Dice for foreground class 1, excluding ignored pixels."""

    def forward(self, predict, target):
        if predict.shape[1] < 2:
            raise ValueError("LandslideDiceLoss requires foreground class 1.")
        logits, labels = _valid_pixels(predict, target, self.ignore_index)
        if not labels.numel():
            return predict.sum() * 0.0
        foreground = logits.softmax(dim=1)[:, 1]
        truth = (labels == 1).to(foreground.dtype)
        intersection = (foreground * truth).sum()
        denominator = foreground.sum() + truth.sum()
        return 1.0 - (2 * intersection + self.smooth) / (denominator + self.smooth)


class BoundaryDiceLoss(nn.Module):
    """Boundary targets are 0/1; augmentation padding uses ignore_index."""

    def __init__(self, ignore_index=255, smooth=1e-5, **kwargs):
        super().__init__()
        self.ignore_index, self.smooth = ignore_index, smooth

    def forward(self, predict, target):
        if predict is None or predict.shape != target.shape:
            raise ValueError("Boundary logits and labels must share shape [B,1,H,W].")
        valid = target != self.ignore_index
        if not valid.any():
            return predict.sum() * 0.0
        truth = target[valid]
        if ((truth != 0) & (truth != 1)).any():
            raise ValueError("Boundary targets must be binary before augmentation.")
        probability = predict[valid].sigmoid()
        intersection = (probability * truth).sum()
        denominator = probability.sum() + truth.sum()
        return 1.0 - (2 * intersection + self.smooth) / (denominator + self.smooth)


class SegmentationCriterion(nn.Module):
    """Shared segmentation objective for train/validation; edge loss is added outside.

    bce_weight is retained as a CLI compatibility name. In ce_dice_boundary it
    weights two-class softmax CE, consistent with the model's two output logits.
    """

    LOSS_TYPES = ("cross_entropy", "focal_loss", "dice_loss", "focal_dice",
                  "focal_dice_boundary", "ce_dice_boundary")

    def __init__(self, loss_type="focal_dice_boundary", bce_weight=0.3,
                 dice_weight=0.8, ignore_index=255):
        super().__init__()
        if loss_type not in self.LOSS_TYPES:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be nonnegative, with positive total.")
        self.loss_type, self.ignore_index = loss_type, ignore_index
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.focal = FocalLoss(ignore_index=ignore_index)
        self.dice = LandslideDiceLoss(ignore_index=ignore_index)

    def forward(self, predict, target):
        if self.loss_type == "cross_entropy":
            return cross_entropy_valid(predict, target, self.ignore_index)
        if self.loss_type == "focal_loss":
            return self.focal(predict, target)
        if self.loss_type == "dice_loss":
            return self.dice(predict, target)
        classification = (cross_entropy_valid(predict, target, self.ignore_index)
                          if self.loss_type == "ce_dice_boundary" else self.focal(predict, target))
        return self.bce_weight * classification + self.dice_weight * self.dice(predict, target)
