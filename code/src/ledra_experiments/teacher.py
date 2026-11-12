from pathlib import Path

import numpy as np
from scipy.special import expit

from .inputs import read, write, sha

LESIONS = ("MA", "HE", "EX", "SE")
LAYERS = ("B06", "B12")


def fit(args):
    from .teacher_probe import fit_probe
    labels = np.load(args.labels, allow_pickle=False)
    if labels.ndim != 3 or labels.shape[1:] != (784, 4) or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Expected original known/unknown IDRiD patch targets [images,784,4]")
    if len(labels) != 44:
        raise ValueError("The recorded teacher uses 44 PROBE_TRAIN images")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("*.npz")):
        raise ValueError("Refusing to overwrite frozen teacher probes")
    outcomes = []
    for layer, path in (("B06", args.features_b06), ("B12", args.features_b12)):
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        if features.shape != (len(labels), 784, 384):
            raise ValueError("Original layer features and teacher training rows disagree")
        membership = np.repeat(np.arange(len(labels)), 784)
        for j, lesion in enumerate(LESIONS):
            known = labels[:, :, j].reshape(-1) >= 0
            fitted = fit_probe(np.asarray(features).reshape(-1, 384)[known], labels[:, :, j].reshape(-1)[known],
                role="PROBE_TRAIN", sample_image_index=membership[known], progress=lambda **row: print(layer, lesion, row, flush=True))
            scaler = fitted["scaler"]
            out = folder / f"{layer}_{lesion}.npz"
            np.savez_compressed(out, coefficients=fitted["coefficients"], bias=fitted["bias"],
                mean=scaler["mean"], scale=scaler["scale"], zero_variance_mask=scaler["zero_variance_mask"])
            outcomes.append(dict(layer=layer, lesion=lesion, artifact=out.name, sha256=sha(out), training=fitted["training_result"]))
    write(folder / "teacher_fit.json", dict(state="completed", labels_sha256=sha(args.labels),
        B06_features_sha256=sha(args.features_b06), B12_features_sha256=sha(args.features_b12),
        scope="Refit from supplied immutable PROBE_TRAIN feature/target order; equality to historical coefficients requires artifact SHA comparison",
        probes=outcomes, source_images_redistributed=False))


def load(folder):
    result = {}
    for layer in LAYERS:
        for lesion in LESIONS:
            with np.load(Path(folder) / f"{layer}_{lesion}.npz", allow_pickle=False) as z:
                probe = {key: z[key].copy() for key in ("mean", "scale", "zero_variance_mask", "coefficients", "bias")}
            if any(probe[key].shape != (384,) for key in ("mean", "scale", "zero_variance_mask", "coefficients")):
                raise ValueError("Frozen teacher dimensionality differs")
            result[layer, lesion] = probe
    return result


def probabilities(features, probes, layer):
    batch = np.asarray(features)
    outputs = []
    for image in batch:
        types = []
        for lesion in LESIONS:
            probe = probes[layer, lesion]
            standardized = (image.astype(np.float64) - probe["mean"]) / probe["scale"]
            standardized[..., probe["zero_variance_mask"]] = 0.
            types.append(expit(standardized @ probe["coefficients"] + float(probe["bias"])))
        outputs.append(types)
    return np.stack(outputs).astype(np.float32)


def extract(args):
    import torch
    from PIL import Image
    from torchvision.transforms import v2
    from lesion_oof.model import build_encoder, GENERAL_WEIGHTS_SHA256
    from .fixed_affine import install
    if not args.device.startswith("cuda"):
        raise ValueError("Original teacher feature arithmetic requires CUDA and Triton")
    if sha(args.weights) != GENERAL_WEIGHTS_SHA256:
        raise ValueError("General encoder weights differ")
    folder = Path(args.data_root) / args.dataset
    rows = read(folder / "rows.json")
    if (folder / "teacher.npy").exists():
        raise ValueError("Refusing to overwrite frozen teacher maps")
    probes = load(args.probes)
    encoder = build_encoder(args.weights).to(args.device).float().eval()
    install(encoder, torch)
    transform = v2.Compose([v2.ToImage(), v2.Resize((392, 392), antialias=True), v2.ToDtype(torch.float32, scale=True)])
    output = np.lib.format.open_memmap(folder / "teacher.npy", mode="w+", dtype=np.float32, shape=(len(rows), 784, 4))
    captured = {}
    hooks = [encoder.blocks[index].register_forward_hook(lambda module, inputs, result, key=layer: captured.__setitem__(key, result[:, 5:].detach().float().cpu().numpy())) for index, layer in ((5, "B06"), (11, "B12"))]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        for lo in range(0, len(rows), args.batch_size):
            rgb = []
            for row in rows[lo:lo + args.batch_size]:
                path = Path(row["image_path"])
                if not path.is_absolute():
                    path = Path(args.image_root) / path
                if row.get("sha256") and sha(path) != row["sha256"]:
                    raise ValueError("Original source image changed")
                with Image.open(path) as im:
                    rgb.append(transform(im.convert("RGB")))
            tensor = (torch.stack(rgb).to(args.device) - .5) / .5
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                encoder.forward_features(tensor)
            targets = np.minimum(probabilities(captured["B06"], probes, "B06"), probabilities(captured["B12"], probes, "B12")).transpose(0, 2, 1)
            output[lo:lo + len(targets)] = targets
            output.flush()
            print(dict(images=lo + len(targets), total_images=len(rows)), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    del output
    write(folder / "teacher_cache_receipt.json", dict(teacher_sha256=sha(folder / "teacher.npy"),
        frozen_probe_sha256={p.name: sha(p) for p in sorted(Path(args.probes).glob("*.npz"))},
        arithmetic="FP32 fixed IEEE Triton affine feature extraction; per-image float64 scaler/linear/sigmoid; float32 minimum B06/B12 maps",
        targets_use_classification_labels=False, original_memberships_required_for_exact_reproduction=True))
