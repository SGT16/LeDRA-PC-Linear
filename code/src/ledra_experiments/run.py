import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from lesion_cross_dataset.metrics import scoring_records, measure_records, source_thresholds
from lesion_exchange.train import seed_all
from lesion_oof.model import build_encoder, GENERAL_WEIGHTS_SHA256, state_hash
from lesion_select.metrics import calibrate
from .factory import BENCHMARK, MATCHED, SEEDS, FOLDS, cache_name, is_published_kernel, make_model
from .inputs import CohortInputs, TokenView, bind_inputs, read, write, sha


def context_digest(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def report(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def runtime(phase, allow_drift=False):
    import timm
    import scipy
    import sklearn
    actual = dict(python=platform.python_version(), torch=str(torch.__version__), timm=timm.__version__, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    expected = dict(python="3.14.6", torch="2.12.1+cu132", timm="1.0.27", numpy="2.4.4", scipy="1.18.0", sklearn="1.9.0") if phase == "matched" else dict(python="3.12.11", torch="2.9.1+cu128", timm="1.0.27", numpy="2.3.4", scipy="1.17.1", sklearn="1.9.0")
    if actual != expected and not allow_drift:
        raise ValueError(f"Recorded runtime differs: expected {expected}; found {actual}. Use --allow-runtime-drift only for explicitly labeled new reruns")
    return dict(actual=actual, expected=expected, recorded_runtime_match=actual == expected)


def initialize(weights, device):
    if sha(weights) != GENERAL_WEIGHTS_SHA256:
        raise ValueError("General pretrained weight file differs from the recorded experiment")
    if device.startswith("cuda") and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("CUDA BF16 is required for the recorded GPU recipe")
    torch.set_num_threads(4)
    return build_encoder(weights).to(device).eval()


def predict(phase, name, model, inputs, indexes):
    if phase == "matched":
        if name == "BASE_MATCHED":
            from lesion_base_matched_v1.worker import predict_base
            return predict_base(model, inputs, indexes)[0]
        from lesion_pc_linear_ablation_v1.worker import predict as score
        return score(model, inputs, indexes)[0]
    if is_published_kernel(name):
        from lesion_published_baselines.train import predict as score
        return score(model, TokenView(inputs), indexes, microbatch=4)
    from lesion_exchange.train import predict as score
    return score(model, inputs, indexes, microbatch=32 if name == "B0" else 8)


@torch.no_grad()
def predict_benchmark_external(name, model, inputs):
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for lo in range(0, len(inputs.rows), 4):
        indexes = np.arange(lo, min(lo + 4, len(inputs.rows)))
        batch = TokenView(inputs).batch(indexes, device) if is_published_kernel(name) else inputs.batch(indexes, device, teacher=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch).float().reshape(-1)
        outputs.append(logits.cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)


def train(args):
    if args.seed not in SEEDS or args.fold not in FOLDS:
        raise ValueError("Use the recorded five seeds and three folds")
    if args.phase == "matched" and not args.device.startswith("cuda"):
        raise ValueError("Matched training requires the recorded CUDA BF16 recipe")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    versions = runtime(args.phase, args.allow_runtime_drift)
    encoder = initialize(args.weights, args.device)
    seed_all(args.seed)
    model = make_model(args.phase, args.model, args.seed, encoder, args.device)
    auxiliary = args.model in ("GLOBAL_RECT", "GLOBAL_LINEAR", "BIDIR_PAIR", "ALIGN_SHUFFLE", "LINEAR_REF", "LINEAR_SHUFFLED", "LINEAR_UNIFORM_REFERENCE", "LINEAR_POOL_ONLY")
    inputs = CohortInputs(args.data_root, args.dataset, cache_name(args.phase, args.model), auxiliary=auxiliary)
    split = next(s for s in inputs.splits if int(s["fold"]) == args.fold)
    context = dict(phase=args.phase, model=args.model, dataset=args.dataset, seed=args.seed,
        fold=args.fold, inputs=bind_inputs(inputs.folder), weights_sha256=sha(args.weights),
        initial_state_sha256=state_hash(model.export_trainable_state()), runtime=versions)
    digest = context_digest(context)
    if (folder / "run.json").exists() and read(folder / "run.json") != context:
        raise ValueError("Output folder belongs to another immutable input/model condition")
    write(folder / "run.json", context)
    if args.phase == "matched":
        if args.model == "BASE_MATCHED":
            from lesion_base_matched_v1.model import verify_initialization_against_saved_reference
            from lesion_pc_linear_ablation_v1.model import make_model as reference_model
            original = reference_model("LINEAR_REF", args.seed, encoder=encoder, device=args.device)
            reference_context = dict(source_seed=args.seed, model=dict(control_id="LINEAR_REF"),
                initial_trainable_state_hash=state_hash(original.export_trainable_state()))
            proof = verify_initialization_against_saved_reference(model, args.seed, encoder, reference_context, args.device)
            proof["scope"] = "Fresh public-run equality; not the absent original per-condition saved context"
            write(folder / "initialization.json", proof)
            del original
            from lesion_base_matched_v1.worker import train_base as fit
        else:
            from lesion_pc_linear_ablation_v1.worker import train as fit
        receipt = fit(model, inputs, split, folder, digest, report, lambda: False)
    elif is_published_kernel(args.model):
        from lesion_published_baselines.train import train_model
        allowed = list(split["fit"]) + list(split["validation"])
        mapping = {old: new for new, old in enumerate(allowed)}
        local_split = dict(fit=[mapping[i] for i in split["fit"]], validation=[mapping[i] for i in split["validation"]])
        config = dict(dataset=args.dataset, seed=args.seed, context_sha256=digest, max_epochs=100,
            patience=20, learning_rate=.001, weight_decay=.0001, original_units_per_batch=16, microbatch=4)
        receipt = train_model(model, TokenView(inputs, allowed), [inputs.rows[i] for i in allowed], local_split,
            folder, config, lambda values: report(**values))
    elif args.model == "B0":
        from importlib import import_module
        module = "lesion_direction_" + args.dataset if args.dataset in ("aptos", "paraguay", "sustech") else "lesion_direction"
        fit = import_module(module + ".train").train_backbone
        config = dict(dataset=args.dataset, seed=args.seed, context_sha256=digest, max_epochs=100,
            patience=20, learning_rate=.001, weight_decay=.0001, microbatch=32, auxiliary_lambda=.1,
            patients_per_batch=16, images_per_batch=16)
        receipt = fit(model, inputs, inputs.rows, split, folder, config, np.full(4, .5, dtype=np.float32),
            lambda values: report(**values))
    else:
        from lesion_allcohorts.train import train_backbone
        config = dict(dataset=args.dataset, seed=args.seed, context_sha256=digest, max_epochs=100,
            patience=20, learning_rate=.001, weight_decay=.0001, microbatch=8, auxiliary_lambda=.1)
        mu = np.asarray(inputs.mu[str(args.fold)], dtype=np.float32) if auxiliary else np.full(4, .5, dtype=np.float32)
        receipt = train_backbone(model, inputs, inputs.rows, split, folder, config, mu,
            lambda values: report(**values))
    inner = predict(args.phase, args.model, model, inputs, split["validation"])
    records = scoring_records(args.dataset, [inputs.rows[i] for i in split["validation"]], inner)
    thresholds = source_thresholds(records, calibrate(records)["t_f1"])
    outer = predict(args.phase, args.model, model, inputs, split["holdout"])
    outer_metrics = measure_records(scoring_records(args.dataset, [inputs.rows[i] for i in split["holdout"]], outer), thresholds)
    np.save(folder / "inner_logits.npy", inner)
    np.save(folder / "outer_logits.npy", outer)
    write(folder / "source_thresholds.json", thresholds)
    write(folder / "outer_metrics.json", outer_metrics)
    write(folder / "completed.json", dict(context_sha256=digest, checkpoint_sha256=sha(folder / "selected_backbone.pt"),
        thresholds_sha256=sha(folder / "source_thresholds.json"), training=receipt, outer=outer_metrics))


def external(args):
    folder = Path(args.fit)
    context, completed = read(folder / "run.json"), read(folder / "completed.json")
    if completed["context_sha256"] != context_digest(context) or completed["checkpoint_sha256"] != sha(folder / "selected_backbone.pt"):
        raise ValueError("Frozen source run binding differs")
    if completed["thresholds_sha256"] != sha(folder / "source_thresholds.json"):
        raise ValueError("Locked source threshold file changed")
    if args.target == context["dataset"]:
        raise ValueError("External target must differ from the training cohort")
    versions = runtime("matched", args.allow_runtime_drift)
    encoder = initialize(args.weights, args.device)
    model = make_model(context["phase"], context["model"], context["seed"], encoder, args.device)
    model.load_trainable_state(torch.load(folder / "selected_backbone.pt", map_location="cpu", weights_only=True))
    target_cache = "prefix0.npy" if context["phase"] == "benchmark" and context["model"] == "LP" else cache_name(context["phase"], context["model"])
    inputs = CohortInputs(args.data_root, args.target, target_cache, auxiliary=False)
    indexes = np.arange(len(inputs.rows))
    logits = predict_benchmark_external(context["model"], model, inputs) if context["phase"] == "benchmark" else predict(context["phase"], context["model"], model, inputs, indexes)
    thresholds = read(folder / "source_thresholds.json")
    result = measure_records(scoring_records(args.target, inputs.rows, logits), thresholds)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "logits.npy", logits)
    write(out / "metrics.json", dict(source_dataset=context["dataset"], target_dataset=args.target,
        model=context["model"], seed=context["seed"], source_fold=context["fold"],
        source_checkpoint_sha256=completed["checkpoint_sha256"], source_thresholds_sha256=completed["thresholds_sha256"],
        target_inputs=bind_inputs(inputs.folder), metrics=result, runtime=versions, target_threshold_tuning=False))
