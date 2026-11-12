import argparse
import csv
from pathlib import Path

from .factory import BENCHMARK, MATCHED, DATASETS, SEEDS, FOLDS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ledra")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--phase", choices=("benchmark", "matched"), required=True)
    plan.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", choices=DATASETS, required=True)
    prepare.add_argument("--rows", required=True)
    prepare.add_argument("--splits")
    prepare.add_argument("--data-root", required=True)
    cache = commands.add_parser("cache")
    teacher_cache = commands.add_parser("teacher-cache")
    for command in (cache, teacher_cache):
        command.add_argument("--dataset", choices=DATASETS, required=True)
        command.add_argument("--data-root", required=True)
        command.add_argument("--image-root", required=True)
        command.add_argument("--weights", required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--batch-size", type=int, default=8)
    teacher_cache.add_argument("--probes", required=True)
    teacher_fit = commands.add_parser("fit-teacher")
    teacher_fit.add_argument("--features-b06", required=True)
    teacher_fit.add_argument("--features-b12", required=True)
    teacher_fit.add_argument("--labels", required=True)
    teacher_fit.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--phase", choices=("benchmark", "matched"), required=True)
    train.add_argument("--model", choices=BENCHMARK + MATCHED, required=True)
    train.add_argument("--dataset", choices=DATASETS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--fold", type=int, choices=FOLDS, required=True)
    train.add_argument("--data-root", required=True)
    train.add_argument("--weights", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--allow-runtime-drift", action="store_true")
    external = commands.add_parser("external")
    external.add_argument("--fit", required=True)
    external.add_argument("--target", choices=DATASETS, required=True)
    external.add_argument("--data-root", required=True)
    external.add_argument("--weights", required=True)
    external.add_argument("--output", required=True)
    external.add_argument("--device", default="cuda")
    external.add_argument("--allow-runtime-drift", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        models = BENCHMARK if args.phase == "benchmark" else MATCHED
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("phase", "model", "dataset", "seed", "fold"))
            writer.writeheader()
            for model in models:
                for dataset in DATASETS:
                    for seed in SEEDS:
                        for fold in FOLDS:
                            writer.writerow(dict(phase=args.phase, model=model, dataset=dataset, seed=seed, fold=fold))
        print(f"{len(models) * 105} training conditions; {len(models) * 630} directed external conditions", flush=True)
    elif args.command == "prepare":
        from .prepare import prepare as call
        call(args)
    elif args.command == "cache":
        from .prepare import extract as call
        call(args)
    elif args.command == "fit-teacher":
        from .teacher import fit as call
        call(args)
    elif args.command == "teacher-cache":
        from .teacher import extract as call
        call(args)
    elif args.command == "train":
        from .run import train as call
        call(args)
    elif args.command == "external":
        from .run import external as call
        call(args)


if __name__ == "__main__":
    main()
