from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Short commands wrapper around train_linear_regression.py for ease of use"
    )
    parser.add_argument(
        "mode",
        choices=["linear", "linear_cat", "rf", "catboost"],
        help=(
            "Which preset to run: "
            "linear (plain linear), "
            "linear_cat (linear + categorical), "
            "rf (random forest + categorical), "
            "catboost (catboost + categorical)."
        ),
    )
    parser.add_argument(
        "--data",
        default="data/realtor-data.zip.csv",
        help="Dataset path (default: data/realtor-data.zip.csv).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster run: caps rows and reduces model complexity (useful for huge CSVs).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to load (forwarded to training script).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for the saved model artifact.",
    )
    args, extra = parser.parse_known_args()

    base = [sys.executable, "train_linear_regression.py", "--data", args.data]

    if args.quick and args.max_rows is None:
        args.max_rows = 50_000

    if args.mode == "linear":
        base += ["--model", "linear"]
    elif args.mode == "linear_cat":
        base += ["--model", "linear", "--include-categorical"]
    elif args.mode == "rf":
        base += ["--model", "rf", "--include-categorical"]
        if args.quick and "--rf-n-estimators" not in extra:
            base += ["--rf-n-estimators", "50"]
    else:
        base += ["--model", "catboost", "--include-categorical"]
        if args.quick and "--cb-iterations" not in extra:
            base += ["--cb-iterations", "500"]
        if args.quick and "--cb-early-stopping" not in extra:
            base += ["--cb-early-stopping", "50"]

    if args.out:
        base += ["--out", args.out]
    if args.max_rows is not None:
        base += ["--max-rows", str(args.max_rows)]

    if extra:
        if extra and extra[0] == "--":
            extra = extra[1:]
        base += extra

    return subprocess.call(base)


if __name__ == "__main__":
    raise SystemExit(main())

