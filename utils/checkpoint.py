"""Shared, explicit checkpoint loading for training and inference."""
from collections.abc import Mapping
import warnings

import numpy as np
import torch


SCHEMA_VERSION = 3
MODEL_CONFIG_KEYS = ("model", "num_classes", "output_stride", "separable_conv")


def load_checkpoint(path, map_location="cpu"):
    """Load tensor checkpoints without silently falling back to arbitrary pickle."""
    if not hasattr(torch.serialization, "safe_globals"):
        raise RuntimeError("Safe checkpoint loading requires a PyTorch version with safe_globals/weights_only support.")
    # Older training logs saved best_score as a NumPy scalar. Allow only these
    # concrete NumPy reconstruction types, not arbitrary checkpoint globals.
    scalar = np._core.multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
    safe_types = [scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))]
    # NumPy 2 renamed numpy.core, but historical checkpoints use the old path.
    if scalar.__module__ != "numpy.core.multiarray":
        safe_types.append((scalar, "numpy.core.multiarray.scalar"))
    with torch.serialization.safe_globals(safe_types):
        return torch.load(path, map_location=map_location, weights_only=True)


def extract_model_state(checkpoint):
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Checkpoint does not contain a model tensor state_dict.")
    return {k.removeprefix("module."): v for k, v in state.items()}


def read_model_config(checkpoint, defaults=None):
    config = dict(defaults or {})
    stored = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
    if not isinstance(stored, Mapping):
        raise ValueError("Checkpoint config must be a dictionary.")
    config.update(stored)
    if checkpoint.get("model_name") is not None:
        if stored.get("model") not in (None, checkpoint["model_name"]):
            raise ValueError("Checkpoint model_name and config.model disagree.")
        config["model"] = checkpoint["model_name"]
    return config


def load_model_weights(model, checkpoint, allow_partial=False):
    state = extract_model_state(checkpoint)
    target = model.module if hasattr(model, "module") else model
    if not allow_partial:
        target.load_state_dict(state, strict=True)
        return {"matched": len(state), "missing": [], "unexpected": [], "skipped": []}
    target_state = target.state_dict()
    matched = {k: v for k, v in state.items() if k in target_state and target_state[k].shape == v.shape}
    if not matched:
        raise ValueError("No checkpoint tensors match this model; refusing empty initialization.")
    missing, unexpected = target.load_state_dict(matched, strict=False)
    skipped = [k for k in state if k not in matched]
    report = {"matched": len(matched), "missing": list(missing), "unexpected": list(unexpected), "skipped": skipped}
    warnings.warn(f"Partial initialization requested: matched={len(matched)}, missing={len(missing)}, skipped={len(skipped)}. Optimizer/training state is not resumed.")
    return report
