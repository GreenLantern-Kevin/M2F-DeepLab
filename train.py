"""Train segmentation with explicit DEM availability and modality dropout."""
import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

import network
from utils.engine import (setup_device_and_seed, resolve_num_classes, build_loaders,
                          build_model, build_metrics, build_optimizer, build_scheduler,
                          build_criterion, save_checkpoint, validate, restore_from_checkpoint,
                          format_seg_metrics_line, is_dual_model)
from utils.create_exp_folder import create_exp_folder
from utils.seg_metrics_logger import SegMetricsLogger
from utils.loss import BoundaryDiceLoss


def gradient_norms(model):
    norms = {}
    for label, fragment in (("dem", "dem_backbone"), ("fhm", ".fhm."), ("mef", ".mef."), ("concat", ".concat.")):
        squares = [p.grad.detach().float().square().sum() for name, p in model.named_parameters()
                   if fragment in name and p.grad is not None]
        norms["grad_" + label] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    return norms


def compute_training_losses(opts, outputs, labels, edges, has_dem, criterion, boundary):
    """All raw components are evaluated once; weights are applied only here."""
    if not isinstance(outputs, dict):
        outputs = {"seg": outputs}
    seg = criterion(outputs["seg"], labels)
    zero = seg.new_zeros(())
    edge = zero
    if outputs.get("edge") is not None and "boundary" in opts.loss_type and opts.alpha > 0:
        edge = boundary(outputs["edge"], edges)
    consist = zero
    if opts.beta > 0 and has_dem.any() and outputs.get("feat_fake") is not None:
        consist = nn.functional.l1_loss(outputs["feat_fake"][has_dem],
                                        outputs["feat_real"][has_dem].detach())
    weighted = {"seg": seg, "edge": opts.alpha * edge,
                "consist": opts.beta * consist}
    total = sum(weighted.values())
    parts = {"loss_seg": seg, "loss_edge": edge, "loss_consist": consist}
    parts.update({"weighted_" + key: value for key, value in weighted.items()})
    return total, parts


def train_one_epoch(opts, model, loader, device, optimizer, scheduler, criterion, epoch):
    model.train()
    if opts.freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    boundary = BoundaryDiceLoss(ignore_index=255).to(device)
    totals = defaultdict(float)
    samples = correct = valid_pixels = grad_batches = 0
    dual = is_dual_model(opts)
    active_dropout = opts.dem_dropout if epoch >= opts.dem_warmup_epochs else 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{opts.epochs}", ncols=110)
    for batch_index, (images, dems, labels, edges, has_dem) in enumerate(progress):
        images, dems = images.to(device, dtype=torch.float32), dems.to(device, dtype=torch.float32)
        labels, edges = labels.to(device, dtype=torch.long), edges.to(device, dtype=torch.float32)
        has_dem = has_dem.to(device, dtype=torch.bool)
        drop_mask = (torch.rand(has_dem.shape, device=device) < active_dropout) & has_dem
        optimizer.zero_grad(set_to_none=True)
        outputs = (model(images, dems, has_dem=has_dem, dem_drop_mask=drop_mask)
                   if dual else model(images))
        loss, parts = compute_training_losses(opts, outputs, labels, edges, has_dem, criterion, boundary)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, batch {batch_index}.")
        loss.backward()
        if batch_index < opts.grad_log_batches:
            for key, value in gradient_norms(model).items():
                totals[key] += value
            grad_batches += 1
        optimizer.step()
        scheduler.step()
        batch_size = images.shape[0]
        samples += batch_size
        totals["train_loss"] += float(loss.detach()) * batch_size
        for key, value in parts.items():
            totals[key] += float(value.detach()) * batch_size
        totals["dem_present"] += int(has_dem.sum())
        if dual:
            totals["route_real"] += int(outputs["used_dem"].sum())
            totals["route_fhm"] += int(outputs["used_fhm"].sum())
            totals["route_zero"] += int((~outputs["used_dem"] & ~outputs["used_fhm"]).sum())
            totals["dem_dropped"] += int(drop_mask.sum())
        prediction = outputs["seg"] if isinstance(outputs, dict) else outputs
        with torch.no_grad():
            valid = labels != 255
            correct += int(((prediction.argmax(1) == labels) & valid).sum())
            valid_pixels += int(valid.sum())
        progress.set_postfix(loss=f"{float(loss.detach()):.3f}", dem=f"{int(has_dem.sum())}/{batch_size}")
    result = {key: value / max(samples, 1) for key, value in totals.items()
              if key.startswith("loss_") or key.startswith("weighted_") or key == "train_loss"}
    result.update({key: int(totals[key]) for key in ("dem_present", "route_real", "route_fhm", "route_zero", "dem_dropped")})
    result.update({key: totals[key] / max(grad_batches, 1) for key in ("grad_dem", "grad_fhm", "grad_mef", "grad_concat")})
    result.update(samples=samples, dem_coverage=totals["dem_present"] / max(samples, 1),
                  train_pacc=correct / max(valid_pixels, 1), dem_dropout_active=active_dropout,
                  alpha=opts.alpha, beta=opts.beta, gradient_batches=grad_batches)
    return result


def main(opts):
    validate_options(opts)
    resolve_num_classes(opts)
    device = setup_device_and_seed(opts)
    train_dst, val_dst, train_loader, val_loader = build_loaders(opts)
    if not len(val_loader) or (not opts.eval_only and not len(train_loader)):
        raise ValueError("Empty loader: check dataset splits and training batch_size/drop_last.")
    opts.total_itrs = opts.epochs * max(len(train_loader), 1)
    opts.eval_resize = opts.val_size
    model = build_model(opts).to(device)
    optimizer = build_optimizer(opts, model)
    scheduler = build_scheduler(opts, optimizer)
    model, best_score, cur_itrs, start_epoch = restore_from_checkpoint(
        opts, model, optimizer, scheduler, train_loader, device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = build_criterion(opts).to(device)
    metrics = build_metrics(opts)
    if opts.eval_only:
        score = validate(opts, model, val_loader, device, metrics, criterion)
        print("[Validation]", format_seg_metrics_line(score))
        if "Paired Missing" in score:
            print("[Paired Real]", format_seg_metrics_line(score["Paired Real"]))
            print("[Paired Missing]", format_seg_metrics_line(score["Paired Missing"]))
        return score

    exp_dir, weights_dir = create_exp_folder()
    config = dict(vars(opts))
    config["normalization"] = {"rgb": "ImageNet", "dem": "ImageNet",
                               "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    config["dem_coverage"] = {"train": train_dst.dem_coverage, "val": val_dst.dem_coverage}
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    logger = SegMetricsLogger(exp_dir)
    print(f"[Train] device={device}, beta={opts.beta}, DEM dropout={opts.dem_dropout}, "
          f"warmup={opts.dem_warmup_epochs}, outputs={exp_dir}")
    for epoch in range(start_epoch, opts.epochs):
        diagnostics = train_one_epoch(opts, model, train_loader, device, optimizer, scheduler, criterion, epoch)
        cur_itrs += len(train_loader)
        score = validate(opts, model, val_loader, device, metrics, criterion, epoch + 1, opts.epochs)
        print("[Validation]", format_seg_metrics_line(score))
        print(f"[DEM] present={diagnostics['dem_present']}/{diagnostics['samples']}, "
              f"real_route={diagnostics['route_real']}, fhm_route={diagnostics['route_fhm']}, zero_route={diagnostics['route_zero']}, "
              f"weighted_consist={diagnostics.get('weighted_consist', 0):.6f}")
        logger.log_epoch(epoch + 1, max(g["lr"] for g in optimizer.param_groups),
                         diagnostics["train_loss"], diagnostics["train_pacc"], score["Loss"], score,
                         diagnostics=diagnostics)
        improved = float(score["Mean IoU"]) > best_score
        best_score = max(best_score, float(score["Mean IoU"]))
        suffix = f"{opts.model}_os{opts.output_stride}_num{opts.num_classes}.pth"
        save_checkpoint(os.path.join(weights_dir, "last_" + suffix), model, optimizer, scheduler,
                        cur_itrs, best_score, opts.model, config=config, epoch=epoch + 1)
        if improved:
            save_checkpoint(os.path.join(weights_dir, "best_" + suffix), model, optimizer, scheduler,
                            cur_itrs, best_score, opts.model, config=config, epoch=epoch + 1)
        logger.save_plots()
    print(f"[Done] Best validation mIoU={best_score:.4f}; outputs={exp_dir}")


def validate_options(opts):
    if not 0 <= opts.dem_dropout <= 1:
        raise ValueError("dem_dropout must be in [0,1].")
    if min(opts.alpha, opts.beta, opts.fusion_init_scale, opts.weight_decay) < 0:
        raise ValueError("Loss weights, fusion_init_scale and weight_decay must be nonnegative.")
    if min(opts.epochs, opts.batch_size, opts.val_batch_size, opts.crop_size, opts.val_size) < 1:
        raise ValueError("Epochs, batch sizes and spatial sizes must be positive.")
    if min(opts.dem_warmup_epochs, opts.grad_log_batches, opts.num_workers) < 0:
        raise ValueError("Warmup, logging count and workers must be nonnegative.")
    if opts.lr <= 0:
        raise ValueError("lr must be positive.")


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="BijieLandslide")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--model", default="m2f_deeplabv3plus_xception16", choices=network.modeling.MODEL_NAMES)
    parser.add_argument("--ckpt", default="", help="Checkpoint used as initialization unless --continue_training.")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--allow_partial_init", action="store_true", help="Explicitly allow partial weight transfer, never resume.")
    parser.add_argument("--loss_type", default="focal_dice_boundary", choices=[
        "cross_entropy", "focal_loss", "dice_loss", "focal_dice", "focal_dice_boundary", "ce_dice_boundary"])
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--separable_conv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True,
                        help="Load backbone initialization only when --ckpt is absent.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lr_policy", default="poly", choices=["poly", "step"])
    parser.add_argument("--step_size", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--val_size", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_encoding", choices=["indices", "binary255"], default="indices")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_missing_dem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bce_weight", type=float, default=0.3, help="CE or Focal loss weight.")
    parser.add_argument("--dice_weight", type=float, default=0.8)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1, help="Boundary loss weight, applied exactly once.")
    parser.add_argument("--beta", type=float, default=0.01, help="FHM paired feature L1 weight; teacher features are detached.")
    parser.add_argument("--dem_dropout", type=float, default=0.2)
    parser.add_argument("--dem_warmup_epochs", type=int, default=5)
    parser.add_argument("--fusion_init_scale", type=float, default=0.1)
    parser.add_argument("--freeze_bn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad_log_batches", type=int, default=1, help="Batches per epoch used to sample branch gradient norms.")
    return parser


if __name__ == "__main__":
    main(get_argparser().parse_args())
