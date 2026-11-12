from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "F1_source_INNER_F1", "accuracy_source_INNER_F1", "Se_at_Sp95",
    "Sp_at_Se95", "AP", "AUROC", "logloss",
)
PRIMARY = METRICS[:4]
COHORTS = ("brset", "deepdrid", "mbrset", "messidor2", "sustech", "paraguay", "aptos")
SEEDS = (823, 289, 592, 605, 329)
FOLDS = (1, 2, 3)
TOLERANCE = 1e-12


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Replay:
    def __init__(self):
        self.numeric_checks = 0
        self.max_absolute_error = 0.0
        self.failures = []
        self.coverage = {}

    def require(self, condition, label):
        if not condition:
            self.failures.append(label)

    def equal(self, observed, expected, label):
        a, b = float(observed), float(expected)
        self.numeric_checks += 1
        difference = abs(a - b)
        self.max_absolute_error = max(self.max_absolute_error, difference)
        if not (math.isfinite(a) and math.isfinite(b) and difference <= TOLERANCE):
            self.failures.append(f"{label}: {a!r} != {b!r}")

    def phase(self, results, phase, external, variants, expected_rows):
        home = results / phase
        prefix = "external" if external else "internal"
        rows = read_csv(home / f"{prefix}_fit_metrics.csv")
        expected = read_csv(home / f"{prefix}_summary.csv")
        identity = ("source_dataset", "target_dataset", "variant") if external else ("dataset", "role", "variant")
        run_key = lambda row: tuple(row[k] for k in identity) + (int(row["seed"]), int(row["fold"]))
        self.require(len(rows) == expected_rows, f"{phase}/{prefix}: row count")
        self.require(len({run_key(row) for row in rows}) == len(rows), f"{phase}/{prefix}: duplicate condition")
        self.require(len({r["variant"] for r in rows}) == variants, f"{phase}/{prefix}: model count")
        if external:
            pairs = {(row["source_dataset"], row["target_dataset"]) for row in rows}
            self.require(pairs == {(s, t) for s in COHORTS for t in COHORTS if s != t}, f"{phase}: all42 directions")
        self.require({int(r["seed"]) for r in rows} == set(SEEDS), f"{phase}/{prefix}: seeds")
        self.require({int(r["fold"]) for r in rows} == set(FOLDS), f"{phase}/{prefix}: folds")
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in identity) + (int(row["seed"]),)].append(row)
        seed_rows = []
        for group, members in sorted(grouped.items()):
            self.require({int(row["fold"]) for row in members} == set(FOLDS) and len(members) == 3, f"{group}: three folds")
            seed = dict(zip(identity, group[:-1]))
            seed["seed"] = group[-1]
            for metric in METRICS:
                seed[metric] = statistics.mean(float(row[metric]) for row in members)
            seed_rows.append(seed)
        per_cell = defaultdict(list)
        for row in seed_rows:
            per_cell[tuple(row[k] for k in identity)].append(row)
        index = {tuple(row[k] for k in identity): row for row in expected}
        self.require(set(per_cell) == set(index), f"{phase}/{prefix}: summary identity coverage")
        summaries = []
        for key, members in sorted(per_cell.items()):
            self.require(len(members) == 5 and {int(row["seed"]) for row in members} == set(SEEDS), f"{key}: five seeds")
            summary = dict(zip(identity, key))
            for metric in METRICS:
                values = [float(row[metric]) for row in members]
                summary[metric + "_mean"] = statistics.mean(values)
                summary[metric + "_seed_SD"] = statistics.stdev(values)
                for suffix in ("_mean", "_seed_SD"):
                    self.equal(summary[metric + suffix], index[key][metric + suffix], f"{phase}/{key}/{metric}{suffix}")
            summaries.append(summary)
        self.coverage[f"{phase}/{prefix}"] = {"fit_role_or_external_records": len(rows), "seed_rows": len(seed_rows), "summary_rows": len(summaries)}
        return {"summaries": summaries, "seed_means": seed_rows, "identity": identity}

    def base_comparison(self, results, phases):
        expected = read_csv(results / "exact_base_comparison" / "comparison_all49.csv")
        model_intervals = read_csv(results / "conditional_intervals" / "model_intervals.csv")
        paired_intervals = read_csv(results / "conditional_intervals" / "paired_intervals.csv")
        interval_index = {(r["test"], r["source_dataset"], r["target_dataset"], r["variant"], r["metric"]): r for r in paired_intervals}
        self.require(len(expected) == 196, "Base comparison: all49 domains x four metrics")
        point_index = {}
        seed_index = defaultdict(list)
        for test, external in (("Internal", False), ("External", True)):
            phase = phases["matched6", external]
            for row in phase["summaries"]:
                if not external and row["role"] != "DEVELOPMENT_OUTER":
                    continue
                source = row["source_dataset"] if external else row["dataset"]
                target = row["target_dataset"] if external else row["dataset"]
                point_index[test, source, target, row["variant"]] = row
            for row in phase["seed_means"]:
                if not external and row["role"] != "DEVELOPMENT_OUTER":
                    continue
                source = row["source_dataset"] if external else row["dataset"]
                target = row["target_dataset"] if external else row["dataset"]
                seed_index[test, source, target, row["variant"]].append(row)
        counts = {scope: {m: {"wins": 0, "ties": 0, "losses": 0} for m in PRIMARY} for scope in ("Internal", "External")}
        for row in expected:
            key = row["test"], row["source_dataset"], row["target_dataset"]
            metric = row["metric"]
            pc = point_index[key + ("LINEAR_REF",)]
            base = point_index[key + ("BASE_MATCHED",)]
            for output, prefix, model in (("PC_mean", "", pc), ("Base_mean", "", base)):
                self.equal(row[output], model[metric + "_mean"], str(key) + output)
            self.equal(row["PC_seed_SD"], pc[metric + "_seed_SD"], str(key) + "PC SD")
            self.equal(row["Base_seed_SD"], base[metric + "_seed_SD"], str(key) + "Base SD")
            delta = pc[metric + "_mean"] - base[metric + "_mean"]
            self.equal(row["delta_pp"], 100 * delta, str(key) + "delta pp")
            a = {int(r["seed"]): r for r in seed_index[key + ("LINEAR_REF",)]}
            b = {int(r["seed"]): r for r in seed_index[key + ("BASE_MATCHED",)]}
            differences = [float(a[s][metric]) - float(b[s][metric]) for s in SEEDS]
            self.equal(row["difference_seed_SD_pp"], 100 * statistics.stdev(differences), str(key) + "paired SD pp")
            ci = interval_index[key + ("BASE_MATCHED", metric)]
            self.equal(row["low95_pp"], 100 * float(ci["low95"]), str(key) + "stored interval lower join")
            self.equal(row["high95_pp"], 100 * float(ci["high95"]), str(key) + "stored interval upper join")
            field = "wins" if delta > TOLERANCE else "losses" if delta < -TOLERANCE else "ties"
            counts[row["test"]][metric][field] += 1
        declared = json.loads((results / "exact_base_comparison" / "comparison_summary.json").read_text(encoding="utf-8"))
        for scope in counts:
            for metric in PRIMARY:
                self.require(counts[scope][metric] == declared["counts"][scope][metric], f"Base {scope}/{metric}: win counts")
        self.require(len(model_intervals) == 49 * 6 * 8, "six models x49 domains x8 model intervals")
        self.require(len(paired_intervals) == 49 * 5 * 8, "five controls x49 domains x8 paired intervals")
        self.coverage["exact_base_comparison"] = {"domains": 49, "four_metric_rows": len(expected), "model_interval_rows": len(model_intervals), "paired_interval_rows": len(paired_intervals), "counts": counts}


def main():
    parser = argparse.ArgumentParser(description="Offline numerical replay from released fit-level aggregate metrics")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--write-recomputed", type=Path, help="Optional directory for newly recomputed numerical summaries")
    args = parser.parse_args()
    replay = Replay()
    checksum_file = args.results / "checksums.sha256"
    hashed = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        file = (args.results / relative).resolve()
        replay.require(file.is_relative_to(args.results.resolve()), "checksum path containment")
        replay.require(file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == digest, "SHA256: " + relative)
        hashed += 1
    phases = {}
    for phase, variants, inner, external in (("benchmark15", 15, 3150, 9450), ("matched6", 6, 1260, 3780)):
        for is_external, number in ((False, inner), (True, external)):
            phases[phase, is_external] = replay.phase(args.results, phase, is_external, variants, number)
            if args.write_recomputed:
                name = phase + ("_external" if is_external else "_internal")
                write_csv(args.write_recomputed / (name + "_summary.csv"), phases[phase, is_external]["summaries"])
                write_csv(args.write_recomputed / (name + "_seed_means.csv"), phases[phase, is_external]["seed_means"])
    replay.base_comparison(args.results, phases)
    table_manifest = json.loads((args.results / "tables" / "tables_manifest.json").read_text(encoding="utf-8"))
    replay.require([r["number"] for r in table_manifest["tables"]] == list(range(1, 20)), "all19 table exports")
    for table in table_manifest["tables"]:
        csv_rows = read_csv(args.results / "tables" / table["file"])
        replay.require(len(csv_rows) == table["rows"], "Table row count: " + str(table["number"]))
    report = {"state": "passed" if not replay.failures else "failed", "hashed_result_files": hashed, "numeric_checks": replay.numeric_checks, "maximum_absolute_error": replay.max_absolute_error, "tolerance": TOLERANCE, "coverage": replay.coverage, "table_exports": 19, "failures": replay.failures, "scope": "Arithmetic replay from per-fit aggregate metrics only; interval endpoints are checked by joins, not re-estimated from individual predictions; no training or independent clinical confirmation."}
    if args.write_recomputed:
        (args.write_recomputed / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(replay.failures))


if __name__ == "__main__":
    main()
