from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DATASET = "ahmedshahriarsakib/usa-real-estate-dataset"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle USA Real Estate Dataset into data/realtor-data.zip.csv"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/realtor-data.zip.csv"),
        help="Output path for the CSV (default: data/realtor-data.zip.csv).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help=f"Kaggle dataset slug (default: {DATASET}).",
    )
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "kaggle CLI not found.\n"
            "Install it with: pip install kaggle\n"
            "Then configure credentials: https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Download + unzip into the output directory.
    _run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            args.dataset,
            "-p",
            str(out_path.parent),
            "--unzip",
            "--force",
        ]
    )

    # Try to find the CSV file that was unzipped.
    candidates = sorted(out_path.parent.glob("*.csv"))
    if not candidates:
        raise SystemExit(f"No .csv files found in {out_path.parent} after download.")

    # Prefer a filename that contains 'realtor', else pick the largest CSV.
    realtor = [p for p in candidates if "realtor" in p.name.lower()]
    chosen = realtor[0] if realtor else max(candidates, key=lambda p: p.stat().st_size)

    if chosen.resolve() != out_path.resolve():
        if out_path.exists():
            out_path.unlink()
        chosen.rename(out_path)

    print(f"downloaded dataset -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


