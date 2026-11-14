import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from ledra_experiments.inputs import sha
from lesion_cross_dataset.metrics import measure_records
from lesion_oof.model import build_encoder, GENERAL_WEIGHTS_SHA256
from .drtid_kernels import MODELS, extract_prefixes, forward_batch, load_model

EYES_SHA256 = "3507e3b5c8c2b8cd2b60c3aed91fa2b98e6412ad7cc93966088cd9b50f1d4f94"
FIELDS_SHA256 = "9354dd1ce0129af2e30c88af1f9dcdf09d78813cbf6ec32472ed436d55d50f0f"


def membership_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def eyes_and_fields(grade_files):
    eyes = []
    used = set()
    for path in grade_files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                macula, disc = row["Macula"].strip(), row["Optic disc"].strip()
                grade = int(row["Grade"])
                if not macula or not disc or macula == disc or used.intersection((macula, disc)) or grade not in range(5):
                    raise ValueError("Repeated field, missing field, or invalid ICDR grade")
                used.update((macula, disc))
                eyes.append((macula, disc, grade))
    counts = np.bincount([grade for _, _, grade in eyes], minlength=5)
    if counts.tolist() != [747, 140, 406, 199, 58] or membership_hash(eyes) != EYES_SHA256:
        raise ValueError("Expected the recorded DRTiD eye pairs, order and ICDR grades")
    return eyes, [name for macula, disc, _ in eyes for name in (macula, disc)]


def image_tensor(root, key):
    from PIL import Image
    from torchvision.transforms import v2
    relative = Path(key if key.lower().endswith(".jpg") else key + ".jpg")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Invalid or missing DRTiD field")
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as photograph:
        rgb = torch.from_numpy(np.array(photograph.convert("RGB"), dtype=np.uint8, copy=True)).permute(2, 0, 1)
    rgb = v2.Resize((392, 392), antialias=True)(rgb)
    rgb = v2.ToDtype(torch.float32, scale=True)(rgb)
    image = v2.Normalize((.5, .5, .5), (.5, .5, .5))(rgb)
    return image, {"key":key, "image_path":path.relative_to(root).as_posix(), "sha256":hashlib.sha256(payload).hexdigest()}


def encode(args, encoder):
    _, fields = eyes_and_fields(args.grades)
    folder = args.cache
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("Use a new cache directory")
    root = args.image_root.resolve()
    shape = (len(fields), 789, 384)
    paths = [folder / name for name in ("preblock.npy", "split11.npy")]
    caches = [np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
              for path, dtype in zip(paths, (np.float16, np.float32))]
    records = []
    for start in range(0, len(fields), 16):
        samples = [image_tensor(root, field) for field in fields[start:start + 16]]
        batch = [image for image, _ in samples]
        records.extend(record for _, record in samples)
        for cache, output in zip(caches, extract_prefixes(encoder, torch.stack(batch))):
            if not np.isfinite(output).all():
                raise ValueError("Nonfinite encoder output")
            cache[start:start + len(batch)] = output
    for cache in caches:
        cache.flush()
    if membership_hash(records) != FIELDS_SHA256:
        raise ValueError("Target images differ from the recorded evaluation")
    binding = {"fields_sha256": FIELDS_SHA256, "grades_sha256": [sha(path) for path in args.grades],
               "cache_sha256": {path.name: sha(path) for path in paths}}
    (folder / "binding.json").write_text(json.dumps(binding), encoding="utf-8")


def eye_metrics(eyes, field_probabilities, thresholds):
    probability = np.asarray(field_probabilities).reshape(len(eyes), 2).max(axis=1)
    label = np.array([int(g >= 2) for _, _, g in eyes])
    records = [{"label": int(y), "probability": float(p)} for y, p in zip(label, probability)]
    metrics = measure_records(records, thresholds)
    if metrics["TP_source_INNER_F1"] + metrics["FP_source_INNER_F1"] == 0:
        metrics["precision_source_INNER_F1"] = None
    metrics["Brier"] = float(np.mean((probability - label) ** 2))
    bins = np.digitize(probability, np.linspace(0, 1, 11)[1:-1])
    metrics["ECE10"] = float(np.abs(np.bincount(bins, weights=probability-label, minlength=10)).sum() / len(label))
    return metrics


def score(args, encoder):
    eyes, fields = eyes_and_fields(args.grades)
    with args.source_locks.open(encoding="utf-8-sig", newline="") as stream:
        fits = [row for row in csv.DictReader(stream)
                if (row["model"], int(row["seed"]), int(row["fold"])) == (args.model, args.seed, args.fold)]
    if len(fits) != 1:
        raise ValueError("Expected exactly one BRSET source lock for the requested model, seed and fold")
    fit = fits[0]
    binding = json.loads((args.cache / "binding.json").read_text(encoding="utf-8"))
    if binding["fields_sha256"] != FIELDS_SHA256 or binding["grades_sha256"] != [sha(path) for path in args.grades]:
        raise ValueError("Bound target images, grades or field order changed")
    filename = "split11.npy" if args.model == "ADAPTERDROP" else "preblock.npy"
    file = args.cache / filename
    if sha(file) != binding["cache_sha256"][filename] or sha(args.checkpoint) != fit["checkpoint_sha256"]:
        raise ValueError("Changed prefix cache or source checkpoint")
    cache = np.load(file, mmap_mode="r", allow_pickle=False)
    if cache.shape != (len(fields), 789, 384):
        raise ValueError("Incomplete target cache")
    model = load_model(encoder, args.model, args.seed, args.checkpoint)
    logits = np.empty(len(fields), dtype=np.float64)
    for start in range(0, len(fields), 4):
        batch = cache[start:start + 4]
        if args.model == "LP":
            batch = forward_batch(model, batch, pool_only=True)
        logits[start:start + 4] = forward_batch(model, batch, pooled_input=args.model == "LP").reshape(-1)
    cuts = {k: fit[k] for k in ("source_INNER_F1", "source_INNER_Se95", "source_INNER_Sp95")}
    metrics = eye_metrics(eyes, expit(logits), cuts)
    report = {"source": "BRSET", "target": "DRTiD", "model": args.model, "seed": args.seed,
              "fold": args.fold, "checkpoint_sha256": fit["checkpoint_sha256"], "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved BRSET benchmark models on the DRTiD dataset")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "score"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--grades", type=Path, nargs=2, required=True)
        cmd.add_argument("--weights", type=Path, required=True)
        cmd.add_argument("--cache", type=Path, required=True)
        if name == "encode":
            cmd.add_argument("--image-root", type=Path, required=True)
        else:
            cmd.add_argument("--model", choices=MODELS, required=True)
            cmd.add_argument("--seed", choices=[823, 289, 592, 605, 329], type=int, required=True)
            cmd.add_argument("--fold", choices=[1, 2, 3], type=int, required=True)
            cmd.add_argument("--checkpoint", type=Path, required=True)
            cmd.add_argument("--source-locks", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if sha(args.weights) != GENERAL_WEIGHTS_SHA256 or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Use the frozen RETFound-Green encoder and a BF16-capable CUDA GPU")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    encoder = build_encoder(args.weights).to("cuda").eval()
    if args.command == "encode":
        encode(args, encoder)
    else:
        score(args, encoder)


if __name__ == "__main__":
    main()
