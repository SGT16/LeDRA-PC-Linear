from __future__ import annotations
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from lesion_exchange.train import seed_all, select_checkpoint, _save_torch, _cpu_state
from lesion_allcohorts.train import batch_indexes, validation_metrics, training_unit
from lesion_oof.io import read_json, write_json, sha

class ArrayInputs:

    def __init__(self, array, indexes=None, features=False):
        self.array = array
        self.indexes = None if indexes is None else self._indexes(indexes, len(array)).copy()
        if self.indexes is not None:
            self.indexes.setflags(write=False)
        self.features = bool(features)

    @staticmethod
    def _indexes(indexes, count):
        values = np.asarray(indexes)
        if values.ndim != 1 or len(values) == 0 or (not np.issubdtype(values.dtype, np.integer)) or (len(set(values.tolist())) != len(values)) or (not bool(((values >= 0) & (values < count)).all())):
            raise ValueError('Invalid, repeated or out-of-view array indexes')
        return values.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.array) if self.indexes is None else len(self.indexes)

    def subset(self, indexes):
        indexes = self._indexes(indexes, len(self))
        mapped = indexes if self.indexes is None else self.indexes[indexes]
        return ArrayInputs(self.array, mapped, self.features)

    def batch(self, indexes, device):
        selected = self._indexes(indexes, len(self))
        if self.indexes is not None:
            selected = self.indexes[selected]
        return torch.as_tensor(np.asarray(self.array[selected]), device=device, dtype=torch.float32)

def forward(model, inputs, indexes, device):
    batch = inputs.batch(indexes, device)
    return (model.forward_features(batch) if inputs.features else model(batch)).float().reshape(-1)

@torch.no_grad()
def predict(model, inputs, indexes, microbatch=4):
    model.eval()
    device = next(model.parameters()).device
    result = []
    for lo in range(0, len(indexes), microbatch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = forward(model, inputs, indexes[lo:lo + microbatch], device)
        result.append(logits.cpu().numpy().astype(np.float64))
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction')
    return values

def chunk_loss(logits, labels, known_count):
    known = (labels == 0) | (labels == 1)
    if known_count <= 0 or logits.shape != labels.shape or int(known.sum()) > known_count:
        raise ValueError('Invalid supervised batch')
    return F.binary_cross_entropy_with_logits(logits[known], labels[known], reduction='sum') / known_count if bool(known.any()) else logits.sum() * 0.0

def train_model(model, inputs, rows, split, folder, config, progress=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if config['max_epochs'] != 100 or config['patience'] != 20 or config['learning_rate'] != 0.001 or (config['weight_decay'] != 0.0001) or (config['original_units_per_batch'] != 16) or (config['microbatch'] != 4):
        raise ValueError('Common comparator protocol changed')
    final, checkpoint, resume = (folder / 'training.json', folder / 'selected_backbone.pt', folder / 'resume.pt')
    if final.exists():
        receipt = read_json(final)
        if receipt['context_sha256'] != config['context_sha256'] or receipt['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('Completed checkpoint binding changed')
        model.load_trainable_state(torch.load(checkpoint, map_location='cpu', weights_only=True))
        return receipt
    seed_all(config['seed'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    device = next(model.parameters()).device
    fit, validation = (list(split['fit']), list(split['validation']))
    if len(fit) != len(set(fit)) or len(validation) != len(set(validation)) or set(fit) & set(validation) or (set(fit + validation) != set(range(len(rows)))):
        raise ValueError('Training receives precisely disjoint FIT and INNER rows')
    labels = np.asarray([int(r['image_label']) if str(r['image_label']) in ('0', '1') else -1 for r in rows])
    validation_rows = [rows[i] for i in validation]
    best, best_state, stale, start_epoch, history = (None, None, 0, 1, [])
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['context_sha256'] != config['context_sha256']:
            raise ValueError('Resume context changed')
        model.load_trainable_state(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        best, best_state, stale = (saved['best'], saved['best_state'], saved['stale'])
        history, start_epoch = (saved['history'], saved['epoch'] + 1)
    for epoch in range(start_epoch, 101):
        if stale >= 20:
            break
        started = time.perf_counter()
        model.train()
        count, loss_sum, batches = (0, 0.0, 0)
        for indexes in batch_indexes(config['dataset'], rows, fit, generator):
            optimizer.zero_grad(set_to_none=True)
            target = torch.as_tensor(labels[indexes], dtype=torch.float32, device=device)
            known_count = int(((target == 0) | (target == 1)).sum())
            if not known_count:
                raise ValueError('Original batch has no own-image labels')
            batch_loss = 0.0
            for lo in range(0, len(indexes), config['microbatch']):
                selected = indexes[lo:lo + config['microbatch']]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits = forward(model, inputs, selected, device)
                loss = chunk_loss(logits, target[lo:lo + len(selected)], known_count)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite classification objective')
                loss.backward()
                batch_loss += float(loss.detach())
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None and (not bool(torch.isfinite(parameter.grad).all())):
                    raise ValueError('Nonfinite comparator gradient')
            optimizer.step()
            count += known_count
            loss_sum += batch_loss * known_count
            batches += 1
        val_logits = predict(model, inputs, validation, config['microbatch'])
        metrics = validation_metrics(config['dataset'], validation_rows, val_logits)
        better, improved = select_checkpoint(metrics, best, epoch)
        stale = 0 if improved else stale + 1
        if better:
            best = dict(epoch=epoch, AP=metrics['AP'], logloss=metrics['logloss'], S5=metrics['S5'])
            best_state = _cpu_state(model)
        item = dict(epoch=epoch, fit_loss=loss_sum / count, known_images=count, batches=batches, validation=metrics, selected=better, stale=stale, seconds=time.perf_counter() - started)
        history.append(item)
        _save_torch(resume, dict(context_sha256=config['context_sha256'], epoch=epoch, model=_cpu_state(model), optimizer=optimizer.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], python_rng=random.getstate(), numpy_rng=np.random.get_state(), best=best, best_state=best_state, stale=stale, history=history))
        write_json(folder / 'history.json', history)
        if progress:
            progress(item)
    if best_state is None:
        raise RuntimeError('No selected model')
    model.load_trainable_state(best_state)
    _save_torch(checkpoint, best_state)
    receipt = dict(state='completed', context_sha256=config['context_sha256'], selected=best, epochs=len(history), max_epochs=100, patience=20, checkpoint_sha256=sha(checkpoint), train_seconds=sum((h['seconds'] for h in history)), selection_role='INNER_VALIDATION', selection_rule='AP; logloss tie break; AP controls patience', scoring_unit=training_unit(config['dataset']), loss='mean BCE on known own-image labels', auxiliary_coefficient=0.0, original_units_per_batch=16, microbatch=4, training_receives_fit_validation_only=True, outer_used_for_training=False)
    write_json(final, receipt)
    return receipt
