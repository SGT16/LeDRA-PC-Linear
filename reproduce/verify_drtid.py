import csv
import itertools
import math
import statistics
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "results" / "drtid"
MODELS = ("LINEAR_REF", "BASE_MATCHED", "LINEAR_POOL_ONLY", "LP", "HOULSBY", "ADAPTFORMER", "ADAPTERPLUS", "ADAPTERDROP")
SEEDS = (823, 289, 592, 605, 329)
FOLDS = (1, 2, 3)
POLICIES = ("source_INNER_F1", "source_INNER_Se95", "source_INNER_Sp95")


def read_index(filename, columns, expected_keys):
    index = {}
    with (DATA / filename).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[column] for column in columns)
            if key in index:
                raise ValueError(f"{filename}: duplicate key {key}")
            index[key] = row
    if set(index) != expected_keys:
        raise ValueError(f"{filename}: unexpected or missing keys")
    return index


def check_value(actual, expected, label):
    if expected is None:
        if actual != "":
            raise ValueError(f"{label}: undefined value must be blank")
        return
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def complete_stat(values, function):
    return None if any(value is None for value in values) else function(values)


def verify():
    seeds = tuple(str(x) for x in SEEDS)
    folds = tuple(str(x) for x in FOLDS)
    with (DATA / "models.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = sorted({row["metric"] for row in csv.DictReader(stream)})
    required = {"Se_at_Sp95", "Sp_at_Se95", "AP", "AUROC", "Brier", "ECE10", "precision_source_INNER_F1"}
    fixed = ("TP", "FP", "FN", "TN", "F1", "accuracy", "sensitivity", "specificity", "FNR", "TN_over_all_n")
    required.update(f"{metric}_{policy}" for metric, policy in itertools.product(fixed, POLICIES))
    if required != set(metrics):
        raise ValueError("Unexpected or missing DRTiD metrics")
    conditions = read_index("conditions.csv", ("model", "seed", "fold"), set(itertools.product(MODELS, seeds, folds)))
    source_locks = read_index("benchmark_source_locks.csv", ("model", "seed", "fold"), set(itertools.product(MODELS[3:], seeds, folds)))
    seed_rows = read_index("seeds.csv", ("model", "seed"), set(itertools.product(MODELS, seeds)))
    model_rows = read_index("models.csv", ("model", "metric"), set(itertools.product(MODELS, metrics)))
    differences = read_index("paired_differences.csv", ("comparator", "metric"), set(itertools.product(MODELS[1:], metrics)))
    for key, record in conditions.items():
        if int(record["eyes"]) != 1550 or int(record["positive_eyes"]) != 663:
            raise ValueError(f"{key}: target cohort counts differ")
        for policy in POLICIES:
            cut = record[policy]
            if cut != "+Infinity" and not 0 <= float(cut) <= 1:
                raise ValueError(f"{key}: invalid source threshold for {policy}")
            if key in source_locks and float(source_locks[key][policy]) != float(cut):
                raise ValueError(f"{key}: threshold differs from original source lock for {policy}")
            tp, fp, fn, tn = [float(record[f"{term}_{policy}"]) for term in ("TP", "FP", "FN", "TN")]
            if not all(value.is_integer() and value >= 0 for value in (tp, fp, fn, tn)):
                raise ValueError(f"{key}: invalid confusion counts")
            if tp + fn != 663 or tn + fp != 887:
                raise ValueError(f"{key}: target population differs")
            check_value(record[f"F1_{policy}"], 2 * tp / (2 * tp + fp + fn), (key, policy, "F1"))
            check_value(record[f"accuracy_{policy}"], (tp + tn) / 1550, (key, policy, "accuracy"))
            expected_rates = {"sensitivity": tp / 663, "specificity": tn / 887,
                              "FNR": fn / 663, "TN_over_all_n": tn / 1550}
            if policy == "source_INNER_F1":
                expected_rates["precision"] = tp / (tp + fp) if tp + fp else None
            for name, value in expected_rates.items():
                check_value(record[f"{name}_{policy}"], value, (key, policy, name))
    means = {}
    for model, metric in itertools.product(MODELS, metrics):
        values = []
        for seed in seeds:
            numbers = [conditions[model, seed, fold][metric] for fold in folds]
            value = complete_stat([float(n) if n != "" else None for n in numbers], statistics.mean)
            check_value(seed_rows[model, seed][metric], value, (model, metric, seed))
            values.append(value)
        row = model_rows[model, metric]
        check_value(row["mean"], complete_stat(values, statistics.mean), (model, metric, "mean"))
        check_value(row["seed_sd"], complete_stat(values, statistics.stdev), (model, metric, "seed_sd"))
        means[model, metric] = values
    for (comparator, metric), row in differences.items():
        values = [None if a is None or b is None else a - b for a, b in zip(means["LINEAR_REF", metric], means[comparator, metric])]
        check_value(row["mean_difference"], complete_stat(values, statistics.mean), (comparator, metric, "mean_difference"))
        check_value(row["seed_sd"], complete_stat(values, statistics.stdev), (comparator, metric, "seed_sd"))
        if None in values:
            if metric != "precision_source_INNER_F1" or any(row[k] for k in ("positive_seeds", "equal_seeds", "negative_seeds")):
                raise ValueError(f"{comparator}, {metric}: incomplete metric or invalid seed counts")
            continue
        counts = (sum(v > 1e-12 for v in values), sum(abs(v) <= 1e-12 for v in values), sum(v < -1e-12 for v in values))
        saved = tuple(int(row[k]) for k in ("positive_seeds", "equal_seeds", "negative_seeds"))
        if saved != counts:
            raise ValueError(f"{comparator}, {metric}: seed direction counts differ")
    print(f"Verified {len(conditions)} evaluations, {len(seed_rows)} seed records, {len(model_rows)} model-metric means and {len(differences)} paired contrasts.")


if __name__ == "__main__":
    verify()
