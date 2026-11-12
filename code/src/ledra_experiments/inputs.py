import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lesion_alignment.model import AlignmentBatch, deterministic_permutations
from lesion_direction.data import fit_mu


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_split(rows, split):
    seen, groups = set(), set()
    for role in ("fit", "validation", "holdout"):
        indexes = list(split[role])
        if not indexes or len(indexes) != len(set(indexes)) or any(type(i) is not int or i < 0 or i >= len(rows) for i in indexes):
            raise ValueError("Invalid partition indexes")
        selected = set(indexes)
        dependency = {str(rows[i]["dependency_group_id"]) for i in indexes}
        if selected & seen or dependency & groups:
            raise ValueError("Partition image/dependency overlap")
        seen |= selected
        groups |= dependency
    if seen != set(range(len(rows))):
        raise ValueError("Partitions must cover the complete cohort")


class CohortInputs:
    def __init__(self, root, dataset, cache, auxiliary=False):
        self.folder = Path(root) / dataset
        self.dataset = dataset
        self.rows = read(self.folder / "rows.json")
        self.splits = read(self.folder / "splits.json")
        if not self.rows or any(row["dataset"] != dataset for row in self.rows):
            raise ValueError("Canonical cohort rows required")
        if {int(s["fold"]) for s in self.splits} != {1, 2, 3}:
            raise ValueError("Exactly three original source folds required")
        for split in self.splits:
            validate_split(self.rows, split)
        self.tokens = np.load(self.folder / cache, mmap_mode="r", allow_pickle=False)
        if len(self.tokens) != len(self.rows) or self.tokens.shape[1:] not in ((789, 384), (384,)):
            raise ValueError("Input cache and canonical rows disagree")
        expected_dtype = np.float16 if cache in ("pc.npy", "prefix0.npy") else np.float32
        if self.tokens.dtype != expected_dtype:
            raise ValueError("Historical cache precision differs")
        self.labels = np.array([int(r["image_label"]) if str(r["image_label"]) in ("0", "1") else -1 for r in self.rows])
        fov_path = self.folder / "fov.npy"
        self.fov = np.load(fov_path, mmap_mode="r", allow_pickle=False) if fov_path.exists() else None
        if self.fov is not None and (self.fov.shape != (len(self.rows), 784) or self.fov.dtype != np.bool_ or not self.fov.any(axis=1).all()):
            raise ValueError("FOV cache differs")
        self.targets = np.load(self.folder / "teacher.npy", mmap_mode="r", allow_pickle=False) if auxiliary else None
        if self.targets is not None and self.targets.shape != (len(self.rows), 784, 4):
            raise ValueError("Teacher cache differs")
        self.permutation = deterministic_permutations(self.rows, self.fov, dataset) if self.fov is not None else None
        self.mu = {str(s["fold"]): fit_mu(self.targets, self.fov, s["fit"]).tolist() for s in self.splits} if auxiliary else {str(s["fold"]): [.5] * 4 for s in self.splits}
        self.features = cache == "features.npy"

    def batch(self, indexes, device, teacher=True):
        ix = np.asarray(indexes, dtype=np.int64)
        tokens = torch.as_tensor(np.asarray(self.tokens[ix]), dtype=torch.float32, device=device)
        if self.tokens.ndim == 2 or self.fov is None:
            return tokens
        return AlignmentBatch(tokens,
            torch.as_tensor(np.asarray(self.targets[ix]), device=device) if teacher and self.targets is not None else None,
            torch.as_tensor(np.asarray(self.fov[ix]), device=device),
            self.permutation[torch.as_tensor(ix)].to(device))


class TokenView:
    def __init__(self, cohort, indexes=None):
        self.cohort = cohort
        self.indexes = np.arange(len(cohort.rows)) if indexes is None else np.asarray(indexes, dtype=np.int64)
        self.features = cohort.features

    def batch(self, indexes, device):
        return torch.as_tensor(np.asarray(self.cohort.tokens[self.indexes[np.asarray(indexes)]]), dtype=torch.float32, device=device)


def bind_inputs(folder):
    folder = Path(folder)
    allowed = ("rows.json", "splits.json", "pc.npy", "prefix0.npy", "prefix2.npy", "prefix6.npy", "prefix10.npy", "prefix11.npy", "features.npy", "fov.npy", "teacher.npy")
    return {name: sha(folder / name) for name in allowed if (folder / name).exists()}
