"""
run_ablation.py — Ablation study driver.

Runs the ML pipeline multiple times, each time toggling off one architectural
or feature component, and collects a comparison table answering:

   "Does each piece of the model actually pull its weight?"

Variants:
  baseline             : full model, no ablation
  no_ticker_embed      : drop the static per-ticker embedding
  no_causal_mask       : allow attention over future timesteps
  no_cs_ranks          : drop cross-sectional rank features
  no_market_feats      : drop market-level features (mkt_ret, mkt_vol, ...)
  last_pooling         : last-token pooling (no attention-weighted pool)
  no_sentiment         : drop FinBERT sentiment features
  no_multi_horizon     : only the 21d target, no 5d/63d heads

Reads each run's MLflow metrics (if MLflow enabled) or the written
artifacts/evaluation_summary.json after each run. Writes
artifacts/ablation_table.csv.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"
ABLATION_DIR = ART_DIR / "ablation_runs"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "baseline",
    "no_ticker_embed",
    "no_causal_mask",
    "no_cs_ranks",
    "no_market_feats",
    "last_pooling",
    "no_sentiment",
    "no_multi_horizon",
]


def run_variant(variant: str, fast: bool = True) -> dict | None:
    cmd = [sys.executable, "run_ml_pipeline.py"]
    if fast:
        cmd.append("--fast")
    cmd += ["--run-name", f"ablation_{variant}"]
    if variant != "baseline":
        cmd += ["--ablation", variant]
    print(f"\n▶ Running variant: {variant}")
    print(f"   cmd: {' '.join(cmd)}")

    t0 = datetime.now()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    dt = (datetime.now() - t0).total_seconds()
    print(f"   runtime: {dt:.1f}s  exit={proc.returncode}")

    # Keep the last 40 lines of stdout for each variant, for debugging
    (ABLATION_DIR / f"{variant}.stdout.txt").write_text(
        "\n".join(proc.stdout.splitlines()[-80:]), encoding="utf-8"
    )
    if proc.returncode != 0:
        (ABLATION_DIR / f"{variant}.stderr.txt").write_text(
            proc.stderr, encoding="utf-8"
        )
        print(f"   ⚠ failed — see {variant}.stderr.txt")
        return None

    # Read the evaluation summary
    summ_path = ART_DIR / "evaluation_summary.json"
    if not summ_path.exists():
        return None
    summ = json.loads(summ_path.read_text(encoding="utf-8"))

    row = {"variant": variant, "runtime_s": round(dt, 1)}
    for model_name in ("baseline_gbr", "transformer"):
        if model_name not in summ:
            continue
        ov = summ[model_name]["overall"][0]
        row[f"{model_name}_ic"] = ov.get("ic")
        row[f"{model_name}_dir_acc"] = ov.get("dir_acc")
        if "monthly_rebalance" in summ[model_name]:
            mr = summ[model_name]["monthly_rebalance"]
            row[f"{model_name}_monthly_sharpe"] = mr.get("sharpe")
            row[f"{model_name}_monthly_total"] = mr.get("total_return")
            row[f"{model_name}_monthly_maxdd"] = mr.get("max_drawdown")
        if summ[model_name]["calibration"]:
            overall_cov = [c for c in summ[model_name]["calibration"]
                           if c["regime"] == "overall"]
            if overall_cov:
                row[f"{model_name}_coverage"] = overall_cov[0]["empirical_coverage"]
                row[f"{model_name}_int_width"] = overall_cov[0]["mean_interval_width"]

    # Preserve a copy of the per-variant summary so runs don't overwrite each other
    (ABLATION_DIR / f"{variant}.summary.json").write_text(
        json.dumps(summ, indent=2, default=str), encoding="utf-8"
    )
    return row


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", default=True,
                   help="use --fast mode for each run (default on — ablation is 8× so speed matters)")
    p.add_argument("--full", action="store_true",
                   help="run in full-mode (slow)")
    p.add_argument("--only", nargs="+", default=None,
                   help="only run specified variants")
    args = p.parse_args()

    fast = not args.full
    variants = args.only or VARIANTS

    rows = []
    for v in variants:
        r = run_variant(v, fast=fast)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No successful variants.")
        return

    df = pd.DataFrame(rows).set_index("variant")
    print("\n" + "=" * 80)
    print("ABLATION TABLE")
    print("=" * 80)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string())

    df.to_csv(ART_DIR / "ablation_table.csv")
    print(f"\n✔ wrote {ART_DIR / 'ablation_table.csv'}")


if __name__ == "__main__":
    main()
