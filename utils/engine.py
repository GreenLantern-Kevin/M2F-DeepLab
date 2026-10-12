"""Shared data, optimization, evaluation, and resumable checkpoint utilities."""
import os
import random

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

import network
import utils
from utils.Landslide import WHULandslideDataset
from utils import m2f_transforms as m2ft
from utils.stream_metrics import StreamSegMetrics
from utils.checkpoint import (SCHEMA_VERSION, MODEL_CONFIG_KEYS, load_checkpoint,
                              load_model_weights, read_model_config)


def setup_device_and_seed(opts):
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_num_classes(opts):
    if opts.num_classes != 2:
        raise ValueError("The landslide dataset requires num_classes=2.")


def is_dual_model(opts):
    return opts.model in network.modeling.DUAL_MODEL_NAMES


def get_dataset(opts):
    normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = m2ft.M2FCompose([
        m2ft.M2FRandomScale((0.8, 1.2)),
        m2ft.M2FRandomCrop(size=opts.crop_size, pad_if_needed=True),
        m2ft.M2FColorJitter(brightness=0.32, contrast=0.5, saturation=0.5, hue=0.1),
        m2ft.M2FRandomHorizontalFlip(), m2ft.M2FToTensor(),
        m2ft.M2FNormalize(**normalize),
    ])
    val_transform = m2ft.M2FCompose([
        m2ft.M2FValResize((opts.val_size, opts.val_size)),
        m2ft.M2FToTensor(), m2ft.M2FNormalize(**normalize),
    ])
    datasets = tuple(WHULandslideDataset(opts.data_root, split=split, transform=transform,
                                       mask_encoding=opts.mask_encoding)
                     for split, transform in (("train", train_transform), ("val", val_transform)))
    for split, dataset in zip(("train", "val"), datasets):
        print(f"[Data] {split} DEM coverage: {dataset.dem_coverage}")
    if is_dual_model(opts) and not opts.eval_only and datasets[0].dem_coverage["present"] == 0:
        raise ValueError("Dual-stream training found no real DEM. Check DEMImages and matching stems.")
    return datasets


def build_loaders(opts):
    train_dst, val_dst = get_dataset(opts)
    common = dict(num_workers=opts.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True,
                                   drop_last=True, **common)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False, **common)
    return train_dst, val_dst, train_loader, val_loader


def build_model(opts):
    kwargs = dict(num_classes=opts.num_classes, output_stride=opts.output_stride,
                  pretrained_backbone=opts.pretrained_backbone and not bool(opts.ckpt))
    if is_dual_model(opts):
        kwargs["fusion_init_scale"] = opts.fusion_init_scale
    model = network.modeling.__dict__[opts.model](**kwargs)
    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    return model


def build_metrics(opts):
    return StreamSegMetrics(opts.num_classes)


def build_optimizer(opts, model):
    """Backbones use 0.1 LR; biases and normalization scales have no decay."""
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        backbone = ("rgb_backbone" in name or "dem_backbone" in name
                    or (not is_dual_model(opts) and name.startswith("backbone.")))
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        key = (backbone, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "param_names": [],
                           "lr": opts.lr * (0.1 if backbone else 1.0),
                           "weight_decay": 0.0 if no_decay else opts.weight_decay,
                           "group_name": f"{'backbone' if backbone else 'head'}_{'no_decay' if no_decay else 'decay'}"}
        groups[key]["params"].append(param)
        groups[key]["param_names"].append(name)
    for group in groups.values():
        print(f"[Optimizer] {group['group_name']}: {len(group['params'])} tensors, "
              f"lr={group['lr']}, weight_decay={group['weight_decay']}")
    return torch.optim.SGD(list(groups.values()), lr=opts.lr, momentum=0.9)


def build_scheduler(opts, optimizer):
    if opts.lr_policy == "poly":
        return utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)


def build_criterion(opts):
    from utils.loss import SegmentationCriterion
    return SegmentationCriterion(opts.loss_type, opts.bce_weight, opts.dice_weight)


def optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(path, model, optimizer, scheduler, cur_itrs, best_score,
                    model_name=None, *, config=None, epoch=None):
    if config is None or epoch is None:
        raise ValueError("A resumable checkpoint requires full config and completed epoch count.")
    unwrapped = model.module if hasattr(model, "module") else model
    np_state = np.random.get_state()
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "model_name": model_name or config["model"],
        "config": dict(config), "epoch": int(epoch), "cur_itrs": int(cur_itrs),
        "best_score": float(best_score), "model_state": unwrapped.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "rng_state": {"torch": torch.get_rng_state(), "python": random.getstate(),
                      "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
                      "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    temporary = str(path) + ".tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def restore_from_checkpoint(opts, model, optimizer, scheduler, train_loader, device=None):
    if not opts.ckpt:
        if opts.continue_training:
            raise ValueError("--continue_training requires --ckpt.")
        return model, 0.0, 0, 0
    checkpoint = load_checkpoint(opts.ckpt)
    stored = read_model_config(checkpoint)
    if opts.continue_training:
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Checkpoint schema is incompatible with this training implementation. "
                             "Use --ckpt without --continue_training to initialize weights, then retrain.")
        required = (*MODEL_CONFIG_KEYS, "fusion_init_scale", "loss_type", "bce_weight", "dice_weight",
                    "alpha", "beta", "dem_dropout", "dem_warmup_epochs",
                    "lr", "weight_decay", "lr_policy", "step_size", "epochs", "total_itrs",
                    "batch_size", "crop_size", "data_root", "mask_encoding", "freeze_bn",
                    "val_size")
        mismatches = [f"{key}: checkpoint={stored.get(key)!r}, requested={getattr(opts, key, None)!r}"
                      for key in required if key not in stored or stored[key] != getattr(opts, key, None)]
        if mismatches:
            raise ValueError("Resume config mismatch:\n" + "\n".join(mismatches))
        saved_groups = checkpoint.get("optimizer_state", {}).get("param_groups", [])
        if [g.get("param_names") for g in saved_groups] != [g["param_names"] for g in optimizer.param_groups]:
            raise ValueError("Optimizer groups changed; initialize weights instead of resuming.")
        completed = int(checkpoint["epoch"])
        cur_itrs = int(checkpoint["cur_itrs"])
        if cur_itrs != completed * len(train_loader):
            raise ValueError("Checkpoint iterations do not match this training loader.")
        load_model_weights(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_state_to_device(optimizer, device or next(model.parameters()).device)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        rng = checkpoint.get("rng_state", {})
        if rng:
            torch.set_rng_state(rng["torch"].cpu())
            random.setstate(rng["python"])
            state = rng["numpy"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
            if torch.cuda.is_available() and rng["cuda"]:
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"[Checkpoint] Resumed after epoch {completed}, iteration {cur_itrs}.")
        return model, float(checkpoint["best_score"]), cur_itrs, completed
    mismatches = [k for k in MODEL_CONFIG_KEYS if k in stored and stored[k] != getattr(opts, k)]
    if mismatches and not opts.allow_partial_init:
        raise ValueError(f"Initialization model config mismatch: {mismatches}. "
                         "For deliberate transfer only, pass --allow_partial_init.")
    report = load_model_weights(model, checkpoint, allow_partial=opts.allow_partial_init)
    print(f"[Checkpoint] Weight initialization: {report}")
    return model, 0.0, 0, 0


def _forward_seg(opts, model, images, dems, has_dem, force_missing=False):
    if is_dual_model(opts):
        outputs = (model(images, None) if force_missing else model(images, dems, has_dem=has_dem))
    else:
        outputs = model(images)
    return outputs["seg"] if isinstance(outputs, dict) else outputs


def predict_probabilities(opts, model, images, dems, has_dem, force_missing=False):
    logits = _forward_seg(opts, model, images, dems, has_dem, force_missing)
    probs = logits.softmax(dim=1)
    return logits, probs


def probabilities_to_masks(probs):
    """Assign every pixel to its highest-probability class."""
    return probs.argmax(dim=1).cpu().numpy().astype(np.uint8)


def validate(opts, model, loader, device, metrics, criterion=None, epoch=None, total_epochs=None):
    metrics.reset()
    paired_real, paired_missing = build_metrics(opts), build_metrics(opts)
    model.eval()
    loss_sum = 0.0
    count = paired_count = present_count = 0
    do_missing = opts.eval_missing_dem and is_dual_model(opts)
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val {epoch}/{total_epochs}" if epoch is not None else "Val", ncols=110)
        for images, dems, labels, edges, has_dem in pbar:
            images, dems = images.to(device), dems.to(device)
            labels, has_dem = labels.to(device, dtype=torch.long), has_dem.to(device, dtype=torch.bool)
            logits, probs = predict_probabilities(opts, model, images, dems, has_dem)
            if criterion is not None:
                loss_sum += float(criterion(logits, labels)) * images.shape[0]
            count += images.shape[0]
            present_count += int(has_dem.sum())
            preds = probabilities_to_masks(probs)
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
            if do_missing and has_dem.any():
                _, fake_probs = predict_probabilities(opts, model, images[has_dem], dems[has_dem],
                                                       has_dem[has_dem], force_missing=True)
                selection = has_dem.cpu().numpy()
                paired_targets = targets[selection]
                paired_real.update(paired_targets, preds[selection])
                paired_missing.update(paired_targets, probabilities_to_masks(fake_probs))
                paired_count += int(has_dem.sum())
            pbar.set_postfix(loss=f"{loss_sum / max(count, 1):.3f}")
    score = metrics.get_results()
    score.update({"Loss": loss_sum / max(count, 1), "DEM Present": present_count, "Samples": count})
    if do_missing:
        score["Paired Samples"] = paired_count
        if paired_count:
            score["Paired Real"] = paired_real.get_results()
            score["Paired Missing"] = paired_missing.get_results()
    return score


def format_seg_metrics_line(score, use_color=True):
    return "  ".join(f"{label}={score.get(key, 0.0):.4f}" for label, key in
                     (("PAcc", "Overall Acc"), ("mAcc", "Mean Acc"), ("mIoU", "Mean IoU"), ("FWIoU", "FreqW Acc")))
