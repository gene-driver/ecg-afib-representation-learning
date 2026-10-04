"""Command-line entry points; imports neural dependencies only when required."""
from argparse import ArgumentParser
from pathlib import Path
import json
import logging

from .config import load_config
from .data import FeatureStore, file_digest, write_json
from .evaluation import compare_runs
from .experiments import run_experiment
from .features import frequency_statistics, prepare_dataset
from .splits import create_splits


def parser():
    root = ArgumentParser(prog="ecg-afib", description="Prepare, train, and compare ECG representation experiments")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build record-aware RR or CWT features from PhysioNet 2017")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--representation", choices=["rr", "wavelet"], default="rr")
    prepare.add_argument("--rr-length", type=int, default=15)
    prepare.add_argument("--rr-mode", choices=["windows", "chunks"], default="windows")
    prepare.add_argument("--frequencies", type=int, default=50)
    prepare.add_argument("--sampling-rate", type=int, default=300)
    prepare.add_argument("--window-samples", type=int, default=4500)
    prepare.add_argument("--cohort", type=Path, help="Existing feature directory whose record IDs/labels must be retained")
    split = commands.add_parser("split", help="Create one stratified recording-level split for all representations")
    split.add_argument("--dataset", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--test-fraction", type=float, default=0.2)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    run = commands.add_parser("run", help="Run an experiment from a TOML preset")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", type=Path, help="Use a new output directory instead of the preset location")
    run.add_argument("--device", help="cpu, cuda, cuda:0, or auto")
    inspect = commands.add_parser("inspect", help="Summarize a prepared dataset")
    inspect.add_argument("--dataset", required=True, type=Path)
    inspect.add_argument("--verify", action="store_true", help="Also check all numeric files and their SHA-256 hashes")
    stats = commands.add_parser("stats", help="Compute shared per-frequency wavelet statistics")
    stats.add_argument("--dataset", required=True, type=Path)
    stats.add_argument("--output", required=True, type=Path)
    compare = commands.add_parser("compare", help="Compare completed runs only when cohort and splits match")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path)
    demo = commands.add_parser("demo", help="Run a synthetic 3D contrastive example; no ECG data required")
    demo.add_argument("--output", type=Path, default=Path("runs/synthetic_demo"))
    demo.add_argument("--epochs", type=int, default=10)
    demo.add_argument("--seed", type=int, default=42)
    return root


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "prepare":
            store = prepare_dataset(args.input, args.output, args.representation, args.rr_length,
                                    args.rr_mode, args.frequencies, args.sampling_rate, args.window_samples, args.cohort)
            result = {"records": len(store.records), "view_shape": store.view_shape, "output": str(args.output)}
        elif args.command == "split":
            result = create_splits(FeatureStore(args.dataset), args.output, args.seed,
                                   args.test_fraction, args.validation_fraction)
        elif args.command == "run":
            config = load_config(args.config)
            if args.output is not None:
                config.paths.output = str(args.output.resolve())
            if args.device:
                config.training.device = args.device
            result = run_experiment(config)
        elif args.command == "inspect":
            store = FeatureStore(args.dataset)
            if args.verify:
                for row in store.records:
                    store.pairs(row["record_id"])
                    if file_digest(store.record_path(row["record_id"])) != row["sha256"]:
                        raise ValueError(f"Feature hash mismatch for {row['record_id']}")
            result = {"records": len(store.records), "normal": sum(r["label"] == 0 for r in store.records),
                      "afib": sum(r["label"] == 1 for r in store.records), "view_shape": store.view_shape,
                      "cohort_hash": store.cohort_hash, "parameters": store.manifest["parameters"], "verified": args.verify}
        elif args.command == "stats":
            store = FeatureStore(args.dataset)
            if store.representation != "wavelet":
                raise ValueError("Frequency statistics require a wavelet feature store")
            result = frequency_statistics(store.pairs(name) for name in store.by_id)
            if args.output.exists():
                raise FileExistsError(f"Statistics file already exists: {args.output}")
            write_json(args.output, result)
        elif args.command == "compare":
            result = compare_runs(args.runs)
            if args.output:
                if args.output.exists():
                    raise FileExistsError(f"Comparison file already exists: {args.output}")
                write_json(args.output, result)
        else:
            from .datasets import run_synthetic_demo
            result = run_synthetic_demo(args.output, args.epochs, args.seed)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        command_parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
