from __future__ import annotations
from importlib import import_module
from pathlib import Path
import time
import numpy as np
import torch
from lesion_exchange.train import seed_all, patient_batches, select_checkpoint, predict, token_batch, _save_torch, _cpu_state
from lesion_direction.data import auxiliary_loss_per_image
from lesion_direction_multiseed.evaluate import metric_records
from lesion_mamba.model import optimizer_for
from lesion_mamba.train import weighted_chunk_loss, AUXILIARY_ARMS
from lesion_oof.io import read_json, write_json, sha
from lesion_select.metrics import summarize
IMAGE_COHORTS = ('sustech', 'paraguay', 'aptos')

def batch_indexes(dataset, rows, fit, generator, units_per_batch=16):
    if units_per_batch != 16:
        raise ValueError('Fixed effective batch contains 16 original units')
    if dataset in IMAGE_COHORTS:
        return import_module('lesion_direction_' + dataset + '.train').image_batches(rows, fit, generator, images_per_batch=16)
    if dataset not in ('brset', 'deepdrid', 'mbrset', 'messidor2'):
        raise ValueError('Unknown original cohort unit')
    return patient_batches(rows, fit, generator, patients_per_batch=16)

def validation_metrics(dataset, rows, logits):
    return summarize(metric_records(dataset, rows, logits))

def training_unit(dataset):
    return 'image' if dataset in IMAGE_COHORTS else 'exam' if dataset == 'messidor2' else 'patient'

def train_backbone(model, inputs, rows, split, folder, config, mu, progress=None):
    dataset = config['dataset']
    unit = training_unit(dataset)
    if config['max_epochs'] != 100 or config['patience'] != 20 or config['microbatch'] != 8 or (config['learning_rate'] != 0.001) or (config['weight_decay'] != 0.0001) or (config['auxiliary_lambda'] != 0.1) or any((r.get('dataset') != dataset for r in rows)):
        raise ValueError('Fixed scientific training recipe or cohort changed')
    if dataset in IMAGE_COHORTS:
        import_module('lesion_direction_' + dataset + '.train')._check_partition(rows, split)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final, selected_path = (folder / 'backbone.json', folder / 'selected_backbone.pt')
    coefficient = config['auxiliary_lambda'] if model.arm in AUXILIARY_ARMS else 0.0
    mu_array = np.asarray(mu.detach().cpu().numpy() if torch.is_tensor(mu) else mu, dtype=np.float32)
    if mu_array.shape != (4,) or not np.isfinite(mu_array).all() or (not ((mu_array > 0) & (mu_array < 1)).all()):
        raise ValueError('Four unchanged fit-only auxiliary frequencies required')
    if final.exists():
        receipt = read_json(final)
        if receipt['context_sha256'] != config['context_sha256'] or receipt['checkpoint_sha256'] != sha(selected_path) or receipt['auxiliary_coefficient'] != coefficient or (receipt['scoring_unit'] != unit) or (receipt['auxiliary_mu'] != mu_array.tolist()):
            raise ValueError('Completed all-cohort checkpoint binding changed')
        model.load_trainable_state(torch.load(selected_path, map_location='cpu', weights_only=True))
        return receipt
    seed_all(config['seed'])
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    device = next(model.parameters()).device
    mu = torch.as_tensor(mu_array, dtype=torch.float32, device=device)
    fit, validation = (split['fit'], split['validation'])
    validation_rows = [rows[int(i)] for i in validation]
    labels = np.asarray([int(r['image_label']) if str(r['image_label']) in ('0', '1') else -1 for r in rows])
    resume = folder / 'resume.pt'
    best, best_state, stale, start_epoch, history = (None, None, 0, 1, [])
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['context_sha256'] != config['context_sha256']:
            raise ValueError('All-cohort resume context changed')
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
        for indexes in batch_indexes(dataset, rows, fit, generator):
            optimizer.zero_grad(set_to_none=True)
            target = torch.as_tensor(labels[indexes], dtype=torch.float32, device=device)
            n = int(((target == 0) | (target == 1)).sum())
            batch_total, batch_class, batch_aux = (0.0, 0.0, 0.0)
            for lo in range(0, len(indexes), config['microbatch']):
                batch = token_batch(inputs, indexes[lo:lo + config['microbatch']], device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits, q = model.forward_training(batch)
                logits = logits.float().reshape(-1)
                auxiliary = None
                if coefficient:
                    if batch.targets is None or batch.fov is None:
                        raise ValueError('Original auxiliary targets and FOV required')
                    auxiliary = auxiliary_loss_per_image(q.float(), batch.targets, batch.fov, mu)
                loss, classification, aux = weighted_chunk_loss(logits, target[lo:lo + len(logits)], auxiliary, coefficient, n, len(indexes))
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite all-cohort objective')
                loss.backward()
                batch_total += float(loss.detach())
                batch_class += float(classification.detach())
                batch_aux += float(aux.detach())
                del loss, classification, aux, logits, q, auxiliary, batch
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None and (not bool(torch.isfinite(parameter.grad).all())):
                    raise ValueError('Nonfinite all-cohort gradient')
            optimizer.step()
            known_count += n
            image_count += len(indexes)
            total_sum += batch_total * n
            class_sum += batch_class * n
            aux_sum += batch_aux * len(indexes)
            batches += 1
        val_logits = predict(model, inputs, validation, config['microbatch'])
        metrics = validation_metrics(dataset, validation_rows, val_logits)
        better, improved = select_checkpoint(metrics, best, epoch)
        stale = 0 if improved else stale + 1
        if better:
            best = dict(epoch=epoch, AP=metrics['AP'], logloss=metrics['logloss'], S5=metrics['S5'])
            best_state = _cpu_state(model)
        item = dict(epoch=epoch, fit_loss=total_sum / known_count, classification_loss=class_sum / known_count, auxiliary_loss=aux_sum / image_count, auxiliary_coefficient=coefficient, images=known_count, all_batch_images=image_count, batches=batches, validation=metrics, selected=better, stale=stale, seconds=time.perf_counter() - started, scoring_unit=unit, original_units_per_batch=16, microbatch=8)
        history.append(item)
        _save_torch(resume, dict(context_sha256=config['context_sha256'], epoch=epoch, model=_cpu_state(model), optimizer=optimizer.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], best=best, best_state=best_state, stale=stale, history=history))
        write_json(folder / 'history.json', history)
        if progress:
            progress(item)
    if best_state is None:
        raise RuntimeError('No selected all-cohort model')
    model.load_trainable_state(best_state)
    _save_torch(selected_path, best_state)
    receipt = dict(state='completed', selection_role='INNER_VALIDATION', selected=best, context_sha256=config['context_sha256'], epochs=len(history), max_epochs=config['max_epochs'], patience=config['patience'], train_seconds=sum((h['seconds'] for h in history)), checkpoint_sha256=sha(selected_path), own_image_targets=True, loss='known-own-image mean BCE plus fixed auxiliary image mean', auxiliary_coefficient=coefficient, gradient_accumulation='immediate weighted microchunk backward; one optimizer step per full original-unit batch', scoring_unit=unit, original_units_per_batch=16, microbatch=8, auxiliary_mu=mu.detach().cpu().tolist(), auxiliary_mu_role='UNCHANGED_ORIGINAL_INNER_FIT_ONLY')
    write_json(final, receipt)
    return receipt
