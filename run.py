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
        default="data/usa_housing.csv",
        help="Dataset path (default: data/usa_housing.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for the saved model artifact.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args to forward to train_linear_regression.py (prefix with --).",
    )
    args = parser.parse_args()

    base = [sys.executable, "train_linear_regression.py", "--data", args.data]

    if args.mode == "linear":
        base += ["--model", "linear"]
    elif args.mode == "linear_cat":
        base += ["--model", "linear", "--include-categorical"]
    elif args.mode == "rf":
        base += ["--model", "rf", "--include-categorical"]
    else:
        base += ["--model", "catboost", "--include-categorical"]

    if args.out:
        base += ["--out", args.out]

    if args.extra:
        extra = args.extra
        if extra and extra[0] == "--":
            extra = extra[1:]
        base += extra

    return subprocess.call(base)


if __name__ == "__main__":
    raise SystemExit(main())

