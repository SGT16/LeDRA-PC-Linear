from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
PATCHES, TYPES, MU_EPSILON = (784, 4, 0.001)

def _arrays(targets, fov):
    targets, fov = (np.asarray(targets), np.asarray(fov))
    if targets.ndim != 3 or targets.shape[1:] != (PATCHES, TYPES) or len(targets) == 0 or (not np.issubdtype(targets.dtype, np.floating)) or (not np.isfinite(targets).all()) or (not ((targets >= 0) & (targets <= 1)).all()):
        raise ValueError('Expected finite soft targets [N,784,4] in [0,1]')
    if fov.shape != targets.shape[:2] or fov.dtype != np.bool_ or (not fov.any(axis=1).all()):
        raise ValueError('Expected a nonempty boolean FOV for every image')
    return (targets, fov)

def fit_mu(targets, fov, fit_indices):
    targets, fov = _arrays(targets, fov)
    indexes = np.asarray(fit_indices)
    if indexes.ndim != 1 or len(indexes) == 0 or (not np.issubdtype(indexes.dtype, np.integer)) or (len(np.unique(indexes)) != len(indexes)) or (indexes < 0).any() or (indexes >= len(targets)).any():
        raise ValueError('Explicit unique nonempty fit row indexes are required')
    mask = fov[indexes]
    value = targets[indexes].astype(np.float64)
    image_means = (value * mask[..., None]).sum(axis=1) / mask.sum(axis=1)[:, None]
    return np.clip(image_means.mean(axis=0), MU_EPSILON, 1 - MU_EPSILON).astype(np.float32)

def auxiliary_loss_per_image(q, targets, fov, mu):
    if not torch.is_tensor(q) or q.ndim != 3 or tuple(q.shape[1:]) != (PATCHES, TYPES) or (len(q) == 0) or (not torch.is_floating_point(q)):
        raise ValueError('Expected semantic logits [B,784,4]')
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = q.float()
        target = torch.as_tensor(targets, device=q.device).detach().float()
        mask = torch.as_tensor(fov, device=q.device).detach()
        average = torch.as_tensor(mu, device=q.device).detach().float()
        if target.shape != logits.shape or mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('Auxiliary target/FOV alignment differs')
        if average.shape != (TYPES,):
            raise ValueError('Fit-only mu must have four entries')
        if not bool(torch.isfinite(logits).all() & torch.isfinite(target).all() & torch.isfinite(average).all()) or not bool(((target >= 0) & (target <= 1)).all()) or (not bool(((average >= MU_EPSILON - 1e-07) & (average <= 1 - MU_EPSILON + 1e-07)).all())) or (not bool(mask.any(dim=1).all())):
            raise ValueError('Invalid finite soft target, clipped mu or empty FOV')
        weight = target / (2 * average) + (1 - target) / (2 * (1 - average))
        weight = weight * mask[..., None]
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return ((loss * weight).sum(dim=1) / weight.sum(dim=1)).mean(dim=1)
