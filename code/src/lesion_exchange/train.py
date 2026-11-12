from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def patient_batches(rows, indexes, generator, patients_per_batch=16):
    bags = {}
    for i in indexes:
        row = rows[int(i)]
        bags.setdefault((row['dataset'], row['patient_id']), []).append(int(i))
    patients = sorted(bags)
    patients = [p for p in patients if any((str(rows[i]['image_label']) in ('0', '1') for i in bags[p]))]
    order = torch.randperm(len(patients), generator=generator).tolist()
    for lo in range(0, len(order), patients_per_batch):
        yield [i for j in order[lo:lo + patients_per_batch] for i in sorted(bags[patients[j]], key=lambda i: rows[i]['image_id'])]

def select_checkpoint(metrics, best, epoch):
    ap, loss = (metrics['AP'], metrics['logloss'])
    if ap is None or not np.isfinite([ap, loss]).all():
        raise ValueError('Invalid validation metrics')
    ap_improved = best is None or ap > best['AP'] + 1e-12
    better = ap_improved or (abs(ap - best['AP']) <= 1e-12 and loss < best['logloss'] - 1e-12)
    return (better, ap_improved)

def token_batch(tokens, indexes, device):
    if hasattr(tokens, 'batch'):
        return tokens.batch(indexes, device)
    if isinstance(tokens, torch.Tensor):
        return tokens[torch.as_tensor(indexes, device=tokens.device)].to(device=device, dtype=torch.float32)
    return torch.as_tensor(tokens[np.asarray(indexes)], dtype=torch.float32, device=device)

@torch.no_grad()
def predict(model, tokens, indexes, microbatch=32):
    model.eval()
    device = next(model.parameters()).device
    result = []
    for lo in range(0, len(indexes), microbatch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            z = model.forward_tokens(token_batch(tokens, indexes[lo:lo + microbatch], device))
        result.append(z.float().reshape(-1).cpu().numpy().astype(np.float64))
    out = np.concatenate(result)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite model output')
    return out

def _save_torch(path, value):
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)

def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.export_trainable_state().items()}

def image_mean_loss(logits, labels):
    known = (labels == 0) | (labels == 1)
    if not bool(known.any()):
        raise ValueError('No known own-image labels')
    return F.binary_cross_entropy_with_logits(logits[known].float(), labels[known].float())
