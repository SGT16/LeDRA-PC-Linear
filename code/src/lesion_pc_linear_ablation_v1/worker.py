from __future__ import annotations
import os
import json, hashlib, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from .model import optimizer_for
from lesion_exchange.train import seed_all, patient_batches, select_checkpoint, _save_torch, _cpu_state
from lesion_mamba.train import weighted_chunk_loss
from lesion_direction.data import auxiliary_loss_per_image
from lesion_select.metrics import summarize
from lesion_cross_dataset.metrics import scoring_records
IMAGE_COHORTS = ('sustech', 'paraguay', 'aptos')

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    for attempt in range(40):
        try:
            tmp.write_text(payload, encoding='utf-8')
            os.replace(tmp, p)
            return
        except PermissionError:
            if attempt == 39:
                if p.name == 'status.json':
                    return
                raise
            time.sleep(0.05)

def require(x, msg):
    if not x:
        raise ValueError(msg)

def batches(inputs, indexes, generator):
    if inputs.dataset not in IMAGE_COHORTS:
        return patient_batches(inputs.rows, indexes, generator, patients_per_batch=16)

    def images():
        ordered = sorted([int(i) for i in indexes], key=lambda i: inputs.rows[i]['source_image_key'])
        require(all((inputs.labels[i] >= 0 for i in ordered)), 'Known own-image targets required for image cohorts')
        order = torch.randperm(len(ordered), generator=generator).tolist()
        for lo in range(0, len(order), 16):
            yield [ordered[j] for j in order[lo:lo + 16]]
    return images()

@torch.no_grad()
def predict(model, inputs, indexes, status=None, phase='scoring', microbatch=8):
    model.eval()
    device = next(model.parameters()).device
    full = []
    base = []
    for lo in range(0, len(indexes), microbatch):
        selected = indexes[lo:lo + microbatch]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            result = model.forward_components(inputs.batch(selected, device, teacher=False))
        full.append(result['z'].float().cpu().numpy().astype(np.float64))
        base.append(result['base'].float().cpu().numpy().astype(np.float64))
        if status:
            status(phase=phase, processed_images=min(lo + microbatch, len(indexes)), total_images=len(indexes))
    z = np.concatenate(full)
    zb = np.concatenate(base)
    require(np.isfinite(z).all() and np.isfinite(zb).all(), 'Nonfinite scoring logits')
    return (z, zb)

def train(model, inputs, split, folder, context, report, pause):
    folder.mkdir(parents=True, exist_ok=True)
    coefficient = model.auxiliary_coefficient
    device = next(model.parameters()).device
    if (folder / 'backbone.json').exists():
        receipt = read(folder / 'backbone.json')
        require(receipt['context_sha256'] == context, 'Completed training context changed')
        require(digest(folder / 'selected_backbone.pt') == receipt['checkpoint_sha256'], 'Selected model changed')
        model.load_trainable_state(torch.load(folder / 'selected_backbone.pt', map_location='cpu', weights_only=True))
        return receipt
    seed = int(model.run_seed)
    seed_all(seed)
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    mu = torch.as_tensor(inputs.mu[str(split['fold'])], dtype=torch.float32, device=device)
    fit = split['fit']
    inner = split['validation']
    inner_rows = [inputs.rows[i] for i in inner]
    best = None
    best_state = None
    stale = 0
    history = []
    start = 1
    resume = folder / 'resume.pt'
    initial_reader = {k: v.clone() for k, v in _cpu_state(model).items() if 'reader' in k}
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        require(saved['context_sha256'] == context, 'Resume context changed')
        model.load_trainable_state(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        best = saved['best']
        best_state = saved['best_state']
        stale = saved['stale']
        history = saved['history']
        start = saved['epoch'] + 1
    for epoch in range(start, 101):
        if stale >= 20:
            break
        started = time.perf_counter()
        model.train()
        known_count = images = 0
        tot = cls = aux = 0.0
        batch_count = 0
        schedule = list(batches(inputs, fit, generator))
        n_batches = len(schedule)
        for step, indexes in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            target = torch.as_tensor(inputs.labels[indexes], dtype=torch.float32, device=device)
            n = int(((target == 0) | (target == 1)).sum())
            bt = bc = ba = 0.0
            for lo in range(0, len(indexes), 8):
                batch = inputs.batch(indexes[lo:lo + 8], device, teacher=bool(coefficient))
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, q = model.forward_training(batch)
                logits = logits.float().reshape(-1)
                a = auxiliary_loss_per_image(q.float(), batch.targets, batch.fov, mu) if coefficient else None
                loss, cl, al = weighted_chunk_loss(logits, target[lo:lo + len(logits)], a, coefficient, n, len(indexes))
                require(bool(torch.isfinite(loss)), 'Nonfinite training objective')
                loss.backward()
                bt += float(loss.detach())
                bc += float(cl.detach())
                ba += float(al.detach())
                del loss, cl, al, logits, q, a, batch
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    require(bool(torch.isfinite(p.grad).all()), 'Nonfinite training gradient')
            optimizer.step()
            known_count += n
            images += len(indexes)
            tot += bt * n
            cls += bc * n
            aux += ba * len(indexes)
            batch_count += 1
            report(phase='training', epoch=epoch, patience=20, stale=stale, step=step, steps=n_batches, processed_images=sum((len(x) for x in schedule[:step])), total_images=sum((len(x) for x in schedule)), last_loss=bt, epoch_elapsed_seconds=time.perf_counter() - started)
        val, _ = predict(model, inputs, inner, lambda **x: report(epoch=epoch, stale=stale, **x), phase='inner_validation')
        inner_records = scoring_records(inputs.dataset, inner_rows, val)
        metrics = summarize(inner_records)
        better, improved = select_checkpoint(metrics, best, epoch)
        stale = 0 if improved else stale + 1
        if better:
            best = dict(epoch=epoch, AP=metrics['AP'], logloss=metrics['logloss'], S5=metrics['S5'])
            best_state = _cpu_state(model)
        item = dict(epoch=epoch, fit_loss=tot / known_count, classification_loss=cls / known_count, auxiliary_loss=aux / images, auxiliary_coefficient=coefficient, images=known_count, all_batch_images=images, batches=batch_count, validation=metrics, selected=better, stale=stale, seconds=time.perf_counter() - started, original_units_per_batch=16, microbatch=8)
        history.append(item)
        _save_torch(resume, dict(context_sha256=context, epoch=epoch, model=_cpu_state(model), optimizer=optimizer.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), best=best, best_state=best_state, stale=stale, history=history))
        write(folder / 'history.json', history)
        report(phase='epoch_saved', epoch=epoch, stale=stale, best_AP=best['AP'], best_epoch=best['epoch'], epoch_seconds=item['seconds'], fit_loss=item['fit_loss'], inner_AP=metrics['AP'], inner_logloss=metrics['logloss'], processed_images=len(inner), total_images=len(inner))
        print(f"{model.control_id} {inputs.dataset} seed{seed} fold{split['fold']} epoch {epoch}/100 patience {stale}/20 AP {metrics['AP']:.6f} time {item['seconds']:.1f}s", flush=True)
        if pause():
            return None
    require(best_state is not None, 'No selected state')
    model.load_trainable_state(best_state)
    if not coefficient:
        final_state = _cpu_state(model)
        require(initial_reader and all((torch.equal(final_state[k], v) for k, v in initial_reader.items())), 'NOAUX readers must remain initialized exactly')
    _save_torch(folder / 'selected_backbone.pt', best_state)
    receipt = dict(state='completed', context_sha256=context, selection_role='INNER_VALIDATION', selected=best, epochs=len(history), max_epochs=100, patience=20, train_seconds=sum((h['seconds'] for h in history)), checkpoint_sha256=digest(folder / 'selected_backbone.pt'), auxiliary_coefficient=coefficient, nominal_parameters=36229, objective_connected_parameters=model.scientific_metadata()['objective_connected_parameters'], training_recipe='Matched native-runtime, FP16 historical block-10 prefix, BF16 suffix/autograd, microbatch8, complete16originalunit batch', auxiliary_mu=mu.cpu().tolist())
    write(folder / 'backbone.json', receipt)
    return receipt
