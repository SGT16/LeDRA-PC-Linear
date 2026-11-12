from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
CONTROL = 'BASE_MATCHED'

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    for attempt in range(40):
        try:
            temporary.write_text(payload, encoding='utf-8')
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 39:
                if path.name == 'status.json':
                    return
                raise
            time.sleep(0.05)

def predict_base(model, inputs, indexes, status=None, phase='scoring', microbatch=8):
    import numpy as np
    import torch
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for lo in range(0, len(indexes), microbatch):
            selected = indexes[lo:lo + microbatch]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                result = model.forward_components(inputs.batch(selected, device, teacher=False))
            require(torch.equal(result['z'], result['base']), 'Independent Base outputs differ')
            outputs.append(result['z'].float().cpu().numpy().astype(np.float64))
            if status:
                status(phase=phase, processed_images=min(lo + microbatch, len(indexes)), total_images=len(indexes))
    result = np.concatenate(outputs)
    require(np.isfinite(result).all(), 'Nonfinite Base scoring logits')
    return (result, result.copy())

def train_base(model, inputs, split, folder, context, report, pause):
    import time
    import torch
    from .model import optimizer_for
    from lesion_pc_linear_ablation_v1.worker import batches
    from lesion_exchange.train import seed_all, select_checkpoint, _save_torch, _cpu_state
    from lesion_mamba.train import weighted_chunk_loss
    from lesion_select.metrics import summarize
    from lesion_cross_dataset.metrics import scoring_records
    require(model.auxiliary_coefficient == 0.0 and model.parameter_counts()['trainable'] == 25377, 'Exact Base training architecture/objective changed')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'backbone.json').exists():
        receipt = read(folder / 'backbone.json')
        require(receipt['context_sha256'] == context and receipt['auxiliary_coefficient'] == 0.0 and (receipt['nominal_parameters'] == 25377), 'Completed Base training context changed')
        require(digest(folder / 'selected_backbone.pt') == receipt['checkpoint_sha256'], 'Selected Base changed')
        model.load_trainable_state(torch.load(folder / 'selected_backbone.pt', map_location='cpu', weights_only=True))
        return receipt
    seed = int(model.run_seed)
    seed_all(seed)
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    device = next(model.parameters()).device
    fit, inner = (split['fit'], split['validation'])
    inner_rows = [inputs.rows[i] for i in inner]
    best, best_state, stale, history, start = (None, None, 0, [], 1)
    resume = folder / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        require(saved['context_sha256'] == context, 'Base resume context changed')
        model.load_trainable_state(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        best, best_state, stale = (saved['best'], saved['best_state'], saved['stale'])
        history, start = (saved['history'], saved['epoch'] + 1)
    for epoch in range(start, 101):
        if stale >= 20:
            break
        started = time.perf_counter()
        model.train()
        known_count = images = batch_count = 0
        total = 0.0
        schedule = list(batches(inputs, fit, generator))
        n_batches = len(schedule)
        for step, indexes in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            target = torch.as_tensor(inputs.labels[indexes], dtype=torch.float32, device=device)
            n = int(((target == 0) | (target == 1)).sum())
            batch_total = 0.0
            for lo in range(0, len(indexes), 8):
                batch = inputs.batch(indexes[lo:lo + 8], device, teacher=False)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits, q = model.forward_training(batch)
                require(q is None, 'Base acquired a semantic reader output')
                logits = logits.float().reshape(-1)
                loss, classification, auxiliary = weighted_chunk_loss(logits, target[lo:lo + len(logits)], None, 0.0, n, len(indexes))
                require(bool(torch.isfinite(loss)), 'Nonfinite Base training objective')
                loss.backward()
                batch_total += float(loss.detach())
                del loss, classification, auxiliary, logits, q, batch
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite Base gradient')
            optimizer.step()
            known_count += n
            images += len(indexes)
            total += batch_total * n
            batch_count += 1
            report(phase='training', epoch=epoch, patience=20, stale=stale, step=step, steps=n_batches, processed_images=sum((len(x) for x in schedule[:step])), total_images=sum((len(x) for x in schedule)), last_loss=batch_total, epoch_elapsed_seconds=time.perf_counter() - started)
        val, _ = predict_base(model, inputs, inner, lambda **x: report(epoch=epoch, stale=stale, **x), phase='inner_validation')
        metrics = summarize(scoring_records(inputs.dataset, inner_rows, val))
        better, improved = select_checkpoint(metrics, best, epoch)
        stale = 0 if improved else stale + 1
        if better:
            best = dict(epoch=epoch, AP=metrics['AP'], logloss=metrics['logloss'], S5=metrics['S5'])
            best_state = _cpu_state(model)
        item = dict(epoch=epoch, fit_loss=total / known_count, classification_loss=total / known_count, auxiliary_loss=0.0, auxiliary_coefficient=0.0, images=known_count, all_batch_images=images, batches=batch_count, validation=metrics, selected=better, stale=stale, seconds=time.perf_counter() - started, original_units_per_batch=16, microbatch=8)
        history.append(item)
        _save_torch(resume, dict(context_sha256=context, epoch=epoch, model=_cpu_state(model), optimizer=optimizer.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], best=best, best_state=best_state, stale=stale, history=history))
        write(folder / 'history.json', history)
        report(phase='epoch_saved', epoch=epoch, stale=stale, best_AP=best['AP'], best_epoch=best['epoch'], epoch_seconds=item['seconds'], fit_loss=item['fit_loss'], inner_AP=metrics['AP'], inner_logloss=metrics['logloss'], processed_images=len(inner), total_images=len(inner))
        print(f"{CONTROL} {inputs.dataset} seed{seed} fold{split['fold']} epoch {epoch}/100 patience {stale}/20 AP {metrics['AP']:.6f} time {item['seconds']:.1f}s", flush=True)
        if pause():
            return None
    require(best_state is not None, 'No selected Base state')
    model.load_trainable_state(best_state)
    _save_torch(folder / 'selected_backbone.pt', best_state)
    receipt = dict(state='completed', context_sha256=context, selection_role='INNER_VALIDATION', selected=best, epochs=len(history), max_epochs=100, patience=20, train_seconds=sum((h['seconds'] for h in history)), checkpoint_sha256=digest(folder / 'selected_backbone.pt'), auxiliary_coefficient=0.0, nominal_parameters=25377, objective_connected_parameters=25377, independent_base_training=True, post_training_branch_zeroing=False, teacher_targets_used=False, training_recipe='Matched native-runtime, FP16 historical block-10 prefix, BF16 suffix/autograd, microbatch8, complete16originalunit batch')
    write(folder / 'backbone.json', receipt)
    return receipt
