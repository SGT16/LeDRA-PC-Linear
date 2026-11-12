from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from lesion_exchange.train import seed_all, patient_batches, image_mean_loss, select_checkpoint, predict, token_batch, _save_torch, _cpu_state
from lesion_oof.io import read_json, write_json, sha
from lesion_oof.model import optimizer_for
from lesion_select.metrics import aggregate, summarize
from .data import auxiliary_loss_per_image
AUXILIARY_ARMS = ('DIR_REAL', 'DIR_SPATIAL', 'AUX_UNTIED')

def compose_loss(logits, labels, auxiliary, coefficient):
    classification = image_mean_loss(logits, labels)
    if coefficient == 0:
        aux = classification.new_zeros(())
    else:
        if coefficient != 0.1 or auxiliary is None or auxiliary.shape != logits.shape:
            raise ValueError('Expected fixed coefficient .1 and one auxiliary loss per image')
        if not bool(torch.isfinite(auxiliary).all()):
            raise ValueError('Nonfinite auxiliary loss')
        aux = auxiliary.float().mean()
    return (classification + coefficient * aux, classification, aux)

def train_backbone(model, inputs, rows, split, folder, config, mu, progress=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final, selected_path = (folder / 'backbone.json', folder / 'selected_backbone.pt')
    coefficient = config['auxiliary_lambda'] if model.arm in AUXILIARY_ARMS else 0.0
    if final.exists():
        receipt = read_json(final)
        if receipt['context_sha256'] != config['context_sha256'] or receipt['checkpoint_sha256'] != sha(selected_path) or receipt['auxiliary_coefficient'] != coefficient:
            raise ValueError('Completed direction checkpoint binding changed')
        model.load_trainable_state(torch.load(selected_path, map_location='cpu', weights_only=True))
        return receipt
    seed_all(config['seed'])
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    device = next(model.parameters()).device
    mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
    fit, validation = (split['fit'], split['validation'])
    validation_rows = [rows[int(i)] for i in validation]
    labels = np.asarray([int(r['image_label']) if str(r['image_label']) in ('0', '1') else -1 for r in rows])
    resume = folder / 'resume.pt'
    best, best_state, stale, start_epoch, history = (None, None, 0, 1, [])
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['context_sha256'] != config['context_sha256']:
            raise ValueError('Direction resume context changed')
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
        known_count, image_count, total_sum, class_sum, aux_sum, batches = (0, 0, 0.0, 0.0, 0.0, 0)
        for indexes in patient_batches(rows, fit, generator, config['patients_per_batch']):
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
                raise ValueError('Nonfinite direction training objective')
            loss.backward()
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None and (not bool(torch.isfinite(parameter.grad).all())):
                    raise ValueError('Nonfinite direction gradient')
            optimizer.step()
            n = int(((target == 0) | (target == 1)).sum())
            known_count += n
            image_count += len(indexes)
            total_sum += float(loss.detach()) * n
            class_sum += float(classification.detach()) * n
            aux_sum += float(aux.detach()) * len(indexes)
            batches += 1
        val_logits = predict(model, inputs, validation, config['microbatch'])
        metrics = summarize(aggregate(validation_rows, val_logits))
        better, improved = select_checkpoint(metrics, best, epoch)
        stale = 0 if improved else stale + 1
        if better:
            best = dict(epoch=epoch, AP=metrics['AP'], logloss=metrics['logloss'], S5=metrics['S5'])
            best_state = _cpu_state(model)
        item = dict(epoch=epoch, fit_loss=total_sum / known_count, classification_loss=class_sum / known_count, auxiliary_loss=aux_sum / image_count, auxiliary_coefficient=coefficient, images=known_count, all_batch_images=image_count, batches=batches, validation=metrics, selected=better, stale=stale, seconds=time.perf_counter() - started)
        history.append(item)
        _save_torch(resume, dict(context_sha256=config['context_sha256'], epoch=epoch, model=_cpu_state(model), optimizer=optimizer.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], best=best, best_state=best_state, stale=stale, history=history))
        write_json(folder / 'history.json', history)
        if progress:
            progress(item)
    if best_state is None:
        raise RuntimeError('No selected direction model')
    model.load_trainable_state(best_state)
    _save_torch(selected_path, best_state)
    receipt = dict(state='completed', selection_role='INNER_VALIDATION', selected=best, context_sha256=config['context_sha256'], epochs=len(history), max_epochs=config['max_epochs'], patience=config['patience'], train_seconds=sum((h['seconds'] for h in history)), checkpoint_sha256=sha(selected_path), own_image_targets=True, loss='known-own-image mean BCE plus fixed auxiliary image mean', auxiliary_coefficient=coefficient, auxiliary_mu=mu.detach().cpu().tolist(), auxiliary_mu_role='INNER_FIT_ONLY')
    write_json(final, receipt)
    return receipt
