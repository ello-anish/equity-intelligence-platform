"""
run_stability.py — Seed-stability driver.

Runs the ML pipeline over N random seeds and reports mean ± std of the key
metrics. Answers the question:

    "Is your single-seed Sharpe of 0.15 a real signal or did the seed just
    land somewhere lucky?"

Typical usage (5 seeds × fast mode ≈ 8 minutes):

    python run_stability.py --seeds 42 7 123 256 999

Writes artifacts/stability_table.csv with mean ± std across seeds.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"
STAB_DIR = ART_DIR / "stability_runs"
STAB_DIR.mkdir(parents=True, exist_ok=True)


def run_seed(seed: int, fast: bool = True) -> dict | None:
    cmd = [sys.executable, "run_ml_pipeline.py",
           "--seed", str(seed), "--run-name", f"seed_{seed}"]
    if fast:
        cmd.append("--fast")
    print(f"\n▶ seed={seed}")
    t0 = datetime.now()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    dt = (datetime.now() - t0).total_seconds()
    print(f"   runtime: {dt:.1f}s  exit={proc.returncode}")
    (STAB_DIR / f"seed_{seed}.stdout.txt").write_text(
        "\n".join(proc.stdout.splitlines()[-80:]), encoding="utf-8"
    )
    if proc.returncode != 0:
        (STAB_DIR / f"seed_{seed}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
        print(f"   ⚠ failed — see seed_{seed}.stderr.txt")
        return None

    summ_path = ART_DIR / "evaluation_summary.json"
    if not summ_path.exists():
        return None
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    (STAB_DIR / f"seed_{seed}.summary.json").write_text(
        json.dumps(summ, indent=2, default=str), encoding="utf-8"
    )

    row = {"seed": seed, "runtime_s": round(dt, 1)}
    for m in ("baseline_gbr", "transformer"):
        if m not in summ:
            continue
        ov = summ[m]["overall"][0]
        row[f"{m}_ic"] = ov.get("ic")
        row[f"{m}_dir_acc"] = ov.get("dir_acc")
        if "monthly_rebalance" in summ[m]:
            row[f"{m}_sharpe"] = summ[m]["monthly_rebalance"].get("sharpe")
            row[f"{m}_total_return"] = summ[m]["monthly_rebalance"].get("total_return")
    return row


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123, 256, 999])
    p.add_argument("--full", action="store_true")
    args = p.parse_args()

    rows = []
    for s in args.seeds:
        r = run_seed(s, fast=not args.full)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No successful runs.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("PER-SEED RESULTS")
    print("=" * 80)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))

    # Summary: mean ± std
    num_cols = [c for c in df.columns if c not in ("seed", "runtime_s") and df[c].dtype.kind in "fi"]
    summary = pd.DataFrame({
        "metric": num_cols,
        "mean": [df[c].mean() for c in num_cols],
        "std": [df[c].std(ddof=1) for c in num_cols],
        "min": [df[c].min() for c in num_cols],
        "max": [df[c].max() for c in num_cols],
    })
    print("\n" + "=" * 80)
    print("STABILITY SUMMARY (N={} seeds)".format(len(df)))
    print("=" * 80)
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))

    df.to_csv(ART_DIR / "stability_per_seed.csv", index=False)
    summary.to_csv(ART_DIR / "stability_summary.csv", index=False)
    print(f"\n✔ wrote {ART_DIR / 'stability_summary.csv'}")


if __name__ == "__main__":
    main()
