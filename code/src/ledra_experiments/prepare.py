import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .factory import DATASETS
from .inputs import read, write, validate_split, sha


def prepare(args):
    path = Path(args.rows)
    rows = read(path) if path.suffix.lower() == ".json" else list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if args.dataset not in DATASETS or not rows or any(row["dataset"] != args.dataset for row in rows):
        raise ValueError("Canonical single-cohort rows required")
    required = {"source_image_key", "image_id", "dependency_group_id", "image_label"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError("Missing required local image metadata")
    if len({row["source_image_key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate canonical source image key")
    folder = Path(args.data_root) / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "rows.json").exists() or (folder / "splits.json").exists():
        raise ValueError("Preparation refuses to overwrite existing memberships")
    if args.splits:
        splits = read(args.splits)
        scope = "Provided immutable original membership indexes"
    else:
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(str(row["dependency_group_id"]), []).append(index)
        keys = sorted(groups)
        strata = np.array([max(int(rows[i].get("patient_label", rows[i]["image_label"])) for i in groups[key]) for key in keys])
        splits = []
        for fold, (allowed, outer) in enumerate(StratifiedKFold(3, shuffle=True, random_state=823).split(keys, strata), 1):
            fit, inner = next(StratifiedShuffleSplit(1, test_size=.2, random_state=823 + 1009 * fold).split(allowed, strata[allowed]))
            expand = lambda selected: sorted(i for group in selected for i in groups[keys[int(group)]])
            splits.append(dict(fold=fold, fit=expand(allowed[fit]), validation=expand(allowed[inner]), holdout=expand(outer)))
        scope = "Regenerated group folds; exact historical identity requires the original canonical row/group order and membership SHA verification"
    if len(splits) != 3:
        raise ValueError("Three fold definitions required")
    for split in splits:
        validate_split(rows, split)
    outer = [index for split in splits for index in split["holdout"]]
    if sorted(outer) != list(range(len(rows))):
        raise ValueError("Every image must occur once in OUTER")
    write(folder / "rows.json", rows)
    write(folder / "splits.json", splits)
    write(folder / "membership_receipt.json", dict(dataset=args.dataset, rows=len(rows),
        dependency_groups=len({str(row["dependency_group_id"]) for row in rows}),
        rows_sha256=sha(folder / "rows.json"), splits_sha256=sha(folder / "splits.json"),
        scope=scope, raw_images_redistributed=False, training_started=False))


def extract(args):
    import torch
    from PIL import Image
    from torchvision.transforms import v2
    from lesion_oof.model import build_encoder, GENERAL_WEIGHTS_SHA256
    from lesion_close_adapters.model import encode_prefix_tokens
    from .inputs import bind_inputs
    if sha(args.weights) != GENERAL_WEIGHTS_SHA256:
        raise ValueError("Frozen general encoder weights differ")
    folder = Path(args.data_root) / args.dataset
    rows = read(folder / "rows.json")
    paths = ["prefix0.npy", "prefix2.npy", "prefix6.npy", "prefix10.npy", "prefix11.npy", "pc.npy", "features.npy", "fov.npy"]
    if any((folder / name).exists() for name in paths):
        raise ValueError("Cache extraction refuses to overwrite existing artifacts")
    device = torch.device(args.device)
    encoder = build_encoder(args.weights).to(device).eval()
    transform = v2.Compose([v2.ToImage(), v2.Resize((392, 392), antialias=True), v2.ToDtype(torch.float32, scale=True)])
    arrays = {}
    for name in paths:
        shape = (len(rows), 384) if name == "features.npy" else (len(rows), 784) if name == "fov.npy" else (len(rows), 789, 384)
        dtype = np.bool_ if name == "fov.npy" else np.float16 if name in ("prefix0.npy", "pc.npy") else np.float32
        arrays[name] = np.lib.format.open_memmap(folder / name, mode="w+", dtype=dtype, shape=shape)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for lo in range(0, len(rows), args.batch_size):
        batch_rows = rows[lo:lo + args.batch_size]
        images = []
        for row in batch_rows:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = Path(args.image_root) / image_path
            if row.get("sha256") and sha(image_path) != row["sha256"]:
                raise ValueError("Original image bytes differ")
            with Image.open(image_path) as im:
                images.append(transform(im.convert("RGB")))
        rgb = torch.stack(images)
        hi = lo + len(rgb)
        fov = rgb.to(torch.float64).reshape(len(rgb), 3, 28, 14, 28, 14).mean(dim=(1, 3, 5)).reshape(len(rgb), 784) > .05
        arrays["fov.npy"][lo:hi] = fov.numpy()
        x = (rgb.to(device) - .5) / .5
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                value = encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(x))))
            arrays["prefix0.npy"][lo:hi] = value.float().cpu().numpy().astype(np.float16)
            raw = encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(x))))
            for block in encoder.blocks[:10]:
                raw = block(raw)
            arrays["pc.npy"][lo:hi] = raw.float().cpu().numpy().astype(np.float16)
            rounded = torch.as_tensor(np.asarray(arrays["prefix0.npy"][lo:hi]).copy(), dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for stage in (2, 6, 10, 11):
                    arrays[f"prefix{stage}.npy"][lo:hi] = encode_prefix_tokens(encoder, rounded, split_block=stage).float().cpu().numpy()
                final = rounded
                for block in encoder.blocks:
                    final = block(final)
                features = encoder.forward_head(encoder.norm(final))
            arrays["features.npy"][lo:hi] = features.float().cpu().numpy()
        for array in arrays.values():
            array.flush()
        print(json.dumps(dict(stage="cache", images=hi, total_images=len(rows))), flush=True)
    del arrays
    write(folder / "cache_receipt.json", dict(weights_sha256=sha(args.weights), artifacts=bind_inputs(folder),
        scope="Regenerated label-blind caches; bit-identical historical payload is established only by matching original artifact hashes",
        teacher_generated=False, training_started=False))
