from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from lesion_exchange.train import seed_all, select_checkpoint, predict, token_batch, _save_torch, _cpu_state
from lesion_direction.train import compose_loss, AUXILIARY_ARMS
from lesion_direction.data import auxiliary_loss_per_image
from lesion_oof.io import read_json, write_json, sha
from lesion_oof.model import optimizer_for
from lesion_select.metrics import summarize
from .metrics import image_records

def image_batches(rows, indexes, generator, images_per_batch=16):
    if images_per_batch != 16:
        raise ValueError('The fixed Paraguay batch contains 16 images')
    indexes = np.asarray(indexes)
    if indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer) or len(indexes) == 0 or (len(set(indexes.tolist())) != len(indexes)) or (not bool(((indexes >= 0) & (indexes < len(rows))).all())):
        raise ValueError('Invalid or repeated fit image indexes')
    ordered = sorted(indexes.tolist(), key=lambda i: rows[i]['source_image_key'])
    keys = [rows[i]['source_image_key'] for i in ordered]
    if len(set(keys)) != len(keys) or any((str(rows[i]['image_label']) not in ('0', '1') for i in ordered)):
        raise ValueError('Fit requires distinct known-label images')
    order = torch.randperm(len(ordered), generator=generator).tolist()
    for lo in range(0, len(order), images_per_batch):
        yield [ordered[j] for j in order[lo:lo + images_per_batch]]

def validation_metrics(rows, logits):
    return summarize(image_records(rows, logits))

def _check_partition(rows, split):
    seen_rows, seen_groups = (set(), set())
    for role in ('fit', 'validation', 'holdout'):
        if role not in split:
            if role == 'holdout':
                continue
            raise ValueError('Missing inner training partition')
        indexes = np.asarray(split[role])
        if indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer) or len(indexes) == 0 or (len(set(indexes.tolist())) != len(indexes)) or (not bool(((indexes >= 0) & (indexes < len(rows))).all())):
            raise ValueError('Invalid image partition')
        ids = set(indexes.tolist())
        groups = {rows[i]['dependency_group_id'] for i in ids}
        if seen_rows & ids or seen_groups & groups:
            raise ValueError('Image/dependency leakage between training partitions')
        seen_rows.update(ids)
        seen_groups.update(groups)
    for i in seen_rows:
        row = rows[i]
        if row.get('dataset') != 'paraguay' or row.get('scoring_unit') != 'image' or row.get('role') != 'TRAIN_FIT' or (row.get('split') != 'TRAIN_FIT') or (str(row.get('image_label')) not in ('0', '1')) or (str(row.get('label')) != str(row.get('image_label'))) or (str(row.get('metric_eligible')) != '1') or ('patient_id' in row) or ('eligible_bilateral' in row):
            raise ValueError('Unexpected Paraguay known-image training scope')

def train_backbone(model, inputs, rows, split, folder, config, mu, progress=None):
    _check_partition(rows, split)
    if config['images_per_batch'] != 16:
        raise ValueError('Expected fixed 16-image minibatches')
    mu_array = np.asarray(mu, dtype=np.float32)
    if mu_array.shape != (4,) or not np.isfinite(mu_array).all() or (not ((mu_array > 0) & (mu_array < 1)).all()):
        raise ValueError('Invalid fit-only auxiliary mu')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final, selected_path = (folder / 'backbone.json', folder / 'selected_backbone.pt')
    coefficient = config['auxiliary_lambda'] if model.arm in AUXILIARY_ARMS else 0.0
    if final.exists():
        receipt = read_json(final)
        if receipt['context_sha256'] != config['context_sha256'] or receipt['checkpoint_sha256'] != sha(selected_path) or receipt['auxiliary_coefficient'] != coefficient or (receipt['scoring_unit'] != 'image') or (receipt['images_per_batch'] != 16) or (receipt['auxiliary_mu'] != mu_array.tolist()):
            raise ValueError('Completed image-unit direction checkpoint binding changed')
        model.load_trainable_state(torch.load(selected_path, map_location='cpu', weights_only=True))
        return receipt
    seed_all(config['seed'])
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    device = next(model.parameters()).device
    mu = torch.as_tensor(mu_array, dtype=torch.float32, device=device)
    fit, validation = (split['fit'], split['validation'])
    validation_rows = [rows[int(i)] for i in validation]
    labels = np.asarray([int(r['image_label']) for r in rows])
    resume = folder / 'resume.pt'
    best, best_state, stale, start_epoch, history = (None, None, 0, 1, [])
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['context_sha256'] != config['context_sha256']:
            raise ValueError('Image-unit direction resume context changed')
        model.load_trainable_state(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        best, best_state, stale = (saved['best'], saved['best_state'], saved['stale'])
        history, start_epoch = (saved['history'], saved['epoch'] + 1)
    for epoch in range(start_epoch, config['max_epochs'] + 1):
        if stale >= config['patience']:
            break
        started = time.perf_counter()
        model.train()
        image_count, total_sum, class_sum, aux_sum, batches = (0, 0.0, 0.0, 0.0, 0)
        for indexes in image_batches(rows, fit, generator, config['images_per_batch']):
            optimizer.zero_grad(set_to_none=True)
            logits_parts, auxiliary_parts = ([], [])
            for lo in range(0, len(indexes), config['microbatch']):
                batch = token_batch(inputs, indexes[lo:lo + config['microbatch']], device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits, q = model.forward_training(batch)
                logits_parts.append(logits.float().reshape(-1))
                if coefficient:
                    if batch.targets is None or batch.fov is None:
                        raise ValueError('Auxiliary training requires explicit frozen targets and FOV')
                    auxiliary_parts.append(auxiliary_loss_per_image(q.float(), batch.targets, batch.fov, mu))
            logits = torch.cat(logits_parts)
            target = torch.as_tensor(labels[indexes], dtype=torch.float32, device=device)
            auxiliary = torch.cat(auxiliary_parts) if auxiliary_parts else None
            loss, classification, aux = compose_loss(logits, target, auxiliary, coefficient)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite image-unit direction objective')
            loss.backward()
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None and (not bool(torch.isfinite(parameter.grad).all())):
                    raise ValueError('Nonfinite image-unit direction gradient')
            optimizer.step()
            n = len(indexes)
            image_count += n
            total_sum += float(loss.detach()) * n
            class_sum += float(classification.detach()) * n
            aux_sum += float(aux.detach()) * n
            batches += 1
        val_logits = predict(model, inputs, validation, config['microbatch'])
        metrics = validation_metrics(validation_rows, val_logits)
        better, improved = select_checkpoint(metrics, best, epoch)
        stale = 0 if improved else stale + 1
        if better:
            best = dict(epoch=epoch, AP=metrics['AP'], logloss=metrics['logloss'], S5=metrics['S5'])
            best_state = _cpu_state(model)
        item = dict(epoch=epoch, fit_loss=total_sum / image_count, classification_loss=class_sum / image_count, auxiliary_loss=aux_sum / image_count, auxiliary_coefficient=coefficient, images=image_count, all_batch_images=image_count, batches=batches, validation=metrics, selected=better, stale=stale, seconds=time.perf_counter() - started, scoring_unit='image')
        history.append(item)
        _save_torch(resume, dict(context_sha256=config['context_sha256'], epoch=epoch, model=_cpu_state(model), optimizer=optimizer.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], best=best, best_state=best_state, stale=stale, history=history))
        write_json(folder / 'history.json', history)
        if progress:
            progress(item)
    if best_state is None:
        raise RuntimeError('No selected image-unit direction model')
    model.load_trainable_state(best_state)
    _save_torch(selected_path, best_state)
    receipt = dict(state='completed', selection_role='INNER_VALIDATION', selected=best, context_sha256=config['context_sha256'], epochs=len(history), max_epochs=config['max_epochs'], patience=config['patience'], train_seconds=sum((h['seconds'] for h in history)), checkpoint_sha256=sha(selected_path), own_image_targets=True, scoring_unit='image', images_per_batch=16, loss='known-image mean BCE plus fixed auxiliary image mean', auxiliary_coefficient=coefficient, auxiliary_mu=mu_array.tolist(), auxiliary_mu_role='INNER_FIT_ONLY')
    write_json(final, receipt)
    return receipt
