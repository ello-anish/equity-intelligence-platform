"""
run_optuna.py — Hyperparameter search for the Transformer, with MLflow logging.

Searches over:
    seq_len        ∈ {30, 45, 60, 90}
    d_model        ∈ {32, 64, 96}
    n_heads        ∈ {2, 4, 8}
    n_layers       ∈ {1, 2, 3}
    dropout        ∈ [0.0, 0.3]
    lr             ∈ [1e-4, 5e-3]  (log)
    pooling        ∈ {"last", "attn"}
    loss_fn        ∈ {"mse", "pairwise"}

Objective: out-of-sample information coefficient (IC) on a single held-out
window. We optimise IC (higher is better) rather than MSE because the
downstream use is ranking, not point prediction.

Every trial logs:
    - trial number, all suggested params
    - train MSE, OOS IC, OOS directional accuracy
    - runtime

to MLflow under the experiment "equity-intel-optuna". After the search,
writes artifacts/optuna_best.json + artifacts/optuna_trials.csv.

Usage:
    python run_optuna.py                # 30 trials, ~25 min
    python run_optuna.py --n-trials 10  # quick sanity
    python run_optuna.py --timeout 600  # cap wall-clock to 10 minutes
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None, help="wall-clock cap in seconds")
    p.add_argument("--fast", action="store_true", default=True,
                   help="use smaller date window for speed (default on)")
    p.add_argument("--full", action="store_true", help="use full data window (slow)")
    return p.parse_args()


def _ic(preds: pd.DataFrame, target_col: str = "target_fwd_ret_21d") -> float:
    """Per-date Spearman IC averaged across dates."""
    if preds.empty:
        return float("nan")
    merged = preds.merge(
        preds[["ticker", "date", target_col]] if target_col in preds.columns else
        preds[["ticker", "date", "target_fwd_ret"]],
        on=["ticker", "date"], how="inner", suffixes=("", "_dup"),
    )
    tcol = target_col if target_col in preds.columns else "target_fwd_ret"
    corrs = []
    for _, grp in preds.groupby("date"):
        if len(grp) < 3:
            continue
        r1 = grp["prediction"].rank()
        r2 = grp[tcol].rank()
        if r1.std() == 0 or r2.std() == 0:
            continue
        corrs.append(float(np.corrcoef(r1, r2)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def build_data(fast: bool):
    from pipeline.ingest import fetch_prices
    from pipeline.quality import run_quality_checks
    from quant.ml.features import build_feature_panel, ALL_FEATURE_COLS

    if fast:
        start, end = "2021-01-01", "2023-06-01"
    else:
        start, end = "2019-01-01", "2024-01-01"
    prices = fetch_prices(start_date=start, end_date=end)
    prices, _ = run_quality_checks(prices)
    panel = build_feature_panel(prices, include_macro=True)
    feature_cols = [c for c in ALL_FEATURE_COLS if c in panel.columns]
    panel = panel.dropna(subset=feature_cols + ["target_fwd_ret_21d"]).reset_index(drop=True)

    # Single train/test split on date — this is NOT the walk-forward split.
    # For hyperparameter search we use a simpler split for speed; whatever
    # we find is validated via the walk-forward CV in the main pipeline.
    dates = panel["date"]
    cutoff_train = dates.quantile(0.7)
    cutoff_val = dates.quantile(0.85)
    train = panel[panel["date"] < cutoff_train].reset_index(drop=True)
    val = panel[(panel["date"] >= cutoff_train) & (panel["date"] < cutoff_val)].reset_index(drop=True)
    test = panel[panel["date"] >= cutoff_val].reset_index(drop=True)
    return panel, feature_cols, train, val, test


def run_search(args):
    try:
        import optuna
    except ImportError:
        print("optuna not installed — install with `pip install optuna`")
        sys.exit(1)

    from quant.ml.tracking import tracker
    from quant.ml.transformer_model import TransformerForecaster, TransformerConfig
    tracker.enable(experiment_name="equity-intel-optuna")

    panel, feature_cols, train, val, test = build_data(fast=not args.full)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  features={len(feature_cols)}")

    def objective(trial):
        cfg = TransformerConfig(
            seq_len=trial.suggest_categorical("seq_len", [30, 45, 60, 90]),
            d_model=trial.suggest_categorical("d_model", [32, 64, 96]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            n_layers=trial.suggest_categorical("n_layers", [1, 2, 3]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=1e-5,
            batch_size=128,
            epochs=4,
            pooling=trial.suggest_categorical("pooling", ["last", "attn"]),
            loss_fn=trial.suggest_categorical("loss_fn", ["mse", "pairwise"]),
            target_cols=("target_fwd_ret_5d", "target_fwd_ret_21d", "target_fwd_ret_63d"),
            primary_target="target_fwd_ret_21d",
        )
        # Ensure d_model divisible by n_heads
        if cfg.d_model % cfg.n_heads != 0:
            raise optuna.TrialPruned(f"d_model={cfg.d_model} not divisible by n_heads={cfg.n_heads}")

        with tracker.run(f"trial_{trial.number}", params={**asdict(cfg)}) as run:
            try:
                forecaster = TransformerForecaster(
                    feature_cols=feature_cols, config=cfg
                ).fit(train)
                preds = forecaster.predict(val, context_df=train)
                if preds.empty:
                    return -999.0
                # Attach target for IC
                merged = preds.merge(
                    val[["ticker", "date", "target_fwd_ret_21d"]],
                    on=["ticker", "date"], how="inner",
                )
                merged["target_fwd_ret"] = merged["target_fwd_ret_21d"]
                ic = _ic(merged, target_col="target_fwd_ret_21d")
                run.log_metric("val_ic", ic)
                # OOS dir acc
                dacc = float((np.sign(merged["prediction"]) == np.sign(merged["target_fwd_ret"])).mean())
                run.log_metric("val_dir_acc", dacc)
                return ic if np.isfinite(ic) else -999.0
            except Exception as e:
                logging.getLogger(__name__).exception("trial %d failed: %s", trial.number, e)
                return -999.0

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   show_progress_bar=False)

    # Persist
    best = {"value": study.best_value, "params": study.best_params}
    (ART_DIR / "optuna_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(ART_DIR / "optuna_trials.csv", index=False)

    print("\n" + "=" * 78)
    print("BEST TRIAL")
    print("=" * 78)
    print(f"val IC: {study.best_value:.4f}")
    print("params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nArtifacts: {ART_DIR / 'optuna_best.json'}, {ART_DIR / 'optuna_trials.csv'}")


if __name__ == "__main__":
    args = parse_args()
    run_search(args)
