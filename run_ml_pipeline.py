"""
run_ml_pipeline.py — End-to-end ML pipeline runner (v2).

What changed from v1:
  ✓ HMM is refit **per fold** on train-only data (no look-ahead leakage)
  ✓ Extended window: 2018-01-01 → 2024-01-01 (COVID crash → 2022 drawdown)
  ✓ Multi-horizon Transformer heads: 5d, 21d, 63d
  ✓ Attention-weighted pooling, attention weights exported to artifacts
  ✓ FinBERT sentiment features (with synthetic-news fallback + cache)
  ✓ Monthly-rebalance long-short backtest (honest non-overlapping Sharpe)
  ✓ Uncertainty-aware allocator using conformal interval widths
  ✓ MLflow experiment tracking (logs params, metrics, artifacts per fold)

Run modes:
  python run_ml_pipeline.py               # full pipeline
  python run_ml_pipeline.py --fast        # smaller window + fewer folds (for dev)
  python run_ml_pipeline.py --no-torch    # skip Transformer (sklearn only)
  python run_ml_pipeline.py --no-sentiment  # skip FinBERT
  python run_ml_pipeline.py --ablation <name>  # run one ablation variant

Artifacts → ./artifacts/
MLflow runs → ./mlruns/
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_pipeline")

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(f" {text}")
    print("=" * 78)


def step(text: str) -> None:
    print()
    print(f"── {text} ".ljust(78, "─"))


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="use smaller window + fewer epochs/folds")
    p.add_argument("--no-torch", action="store_true", help="skip Transformer model")
    p.add_argument("--no-sentiment", action="store_true", help="skip FinBERT sentiment features")
    p.add_argument("--no-macro", action="store_true", help="skip real macro features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--tc-bps", type=float, default=20.0,
                   help="transaction cost in bps for backtest/allocator (default 20)")
    p.add_argument("--rank-loss", action="store_true",
                   help="train Transformer with pairwise-ranking loss")
    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=[None, "no_ticker_embed", "no_causal_mask", "no_cs_ranks",
                 "no_market_feats", "no_macro_feats",
                 "last_pooling", "attn_pooling",
                 "no_sentiment", "no_multi_horizon", "mse_loss", "pairwise_loss"],
        help="one-at-a-time ablation variant",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    t_start = datetime.now()

    # ── Pipeline configuration ────────────────────────────────────────────
    if args.fast:
        START_DATE, END_DATE = "2020-01-01", "2023-01-01"
        EPOCHS = 4
        TRAIN_WINDOW = 365
        TEST_WINDOW = 63
        STEP = 63
    else:
        # Extended window: covers COVID crash (Mar 2020), 2022 drawdown, 2023 recovery.
        # step=126 (~6mo) keeps fold count manageable for laptop runtime (~10-15 min).
        START_DATE, END_DATE = "2019-01-01", "2024-01-01"
        EPOCHS = 6
        TRAIN_WINDOW = 504
        TEST_WINDOW = 63
        STEP = 126
    SEQ_LEN = 60
    EMBARGO = 21

    have_torch = not args.no_torch
    try:
        if have_torch:
            import torch  # noqa: F401
    except Exception as e:
        logger.warning("torch unavailable (%s) — skipping Transformer", e)
        have_torch = False

    # ── MLflow ────────────────────────────────────────────────────────────
    from quant.ml.tracking import tracker
    run_name = args.run_name or ("ablation_" + args.ablation if args.ablation else "baseline_run")
    tracker.enable(experiment_name="equity-intelligence")

    banner(f"EQUITY INTELLIGENCE PLATFORM — ML PIPELINE v2  (run: {run_name})")
    print(f"Config: {START_DATE} → {END_DATE}, train={TRAIN_WINDOW}d, test={TEST_WINDOW}d, "
          f"embargo={EMBARGO}d, step={STEP}d, seq_len={SEQ_LEN}, seed={args.seed}")

    with tracker.run(run_name, params={
        "start_date": START_DATE, "end_date": END_DATE, "seed": args.seed,
        "train_window": TRAIN_WINDOW, "test_window": TEST_WINDOW,
        "embargo": EMBARGO, "step": STEP, "seq_len": SEQ_LEN,
        "epochs": EPOCHS, "have_torch": have_torch,
        "ablation": args.ablation or "none",
        "sentiment": not args.no_sentiment,
    }, tags={"pipeline_version": "v2"}) as run:

        # ── 1. Data ──────────────────────────────────────────────────────
        step("Step 1  Fetching prices")
        from pipeline.ingest import fetch_prices
        from pipeline.quality import run_quality_checks
        prices = fetch_prices(start_date=START_DATE, end_date=END_DATE)
        prices, qrep = run_quality_checks(prices)
        print(f"  rows={len(prices)}  tickers={prices['ticker'].nunique()}")
        run.log_metric("n_price_rows", len(prices))
        run.log_metric("n_tickers", prices["ticker"].nunique())

        # ── 2. Sentiment features (optional) ─────────────────────────────
        sentiment_df = None
        if not args.no_sentiment and args.ablation != "no_sentiment":
            step("Step 2  Computing sentiment features (FinBERT)")
            from quant.ml.sentiment import compute_sentiment_features
            try:
                sentiment_df = compute_sentiment_features(prices, use_cache=True)
                print(f"  sentiment rows: {len(sentiment_df)}")
                run.log_metric("n_sentiment_rows", len(sentiment_df))
            except Exception as e:
                logger.warning("Sentiment layer failed (%s) — continuing without it", e)
                sentiment_df = None

        # ── 3. Feature engineering ───────────────────────────────────────
        step("Step 3  Building feature panel")
        from quant.ml.features import (
            build_feature_panel, FEATURE_COLS, MACRO_FEATURE_COLS,
        )
        from quant.ml.sentiment import merge_sentiment_onto_panel, SENTIMENT_FEATURE_COLS

        include_macro = not args.no_macro and args.ablation != "no_macro_feats"
        panel = build_feature_panel(prices, include_macro=include_macro)
        all_features: List[str] = list(FEATURE_COLS)
        if include_macro:
            all_features += [c for c in MACRO_FEATURE_COLS if c in panel.columns]
        if sentiment_df is not None and not sentiment_df.empty:
            panel = merge_sentiment_onto_panel(panel, sentiment_df)
            all_features += [c for c in SENTIMENT_FEATURE_COLS if c in panel.columns]

        # Ablation: drop certain feature groups
        if args.ablation == "no_cs_ranks":
            all_features = [c for c in all_features if not c.startswith("cs_rank_")]
        elif args.ablation == "no_market_feats":
            all_features = [c for c in all_features if not c.startswith("mkt_")]

        print(f"  panel={panel.shape}  features={len(all_features)}")
        run.log_metric("n_panel_rows", len(panel))
        run.log_metric("n_features", len(all_features))
        panel.to_parquet(ART_DIR / "feature_panel.parquet", index=False)

        # ── 4. Purged walk-forward CV ────────────────────────────────────
        step("Step 4  Purged walk-forward CV (HMM refit per fold)")
        from quant.ml.walkforward import PurgedWalkForward
        from quant.ml.baseline import BaselineForecaster
        from quant.ml.conformal import SplitConformalWrapper
        from quant.ml.regimes import fit_regime_model

        if have_torch:
            from quant.ml.transformer_model import TransformerForecaster, TransformerConfig

        wf = PurgedWalkForward(TRAIN_WINDOW, TEST_WINDOW, EMBARGO, STEP)
        fold_plan = wf.describe(panel["date"])
        fold_plan.to_csv(ART_DIR / "fold_plan.csv", index=False)
        print(fold_plan.to_string(index=False))
        run.log_metric("n_folds", len(fold_plan))

        baseline_preds = []
        transformer_preds = []
        regime_frames = []
        attention_samples = []

        panel_idx = panel.reset_index(drop=True)
        for split in wf.split(panel_idx["date"]):
            logger.info("fold %d  train=%s→%s  test=%s→%s",
                        split.fold, split.train_start.date(), split.train_end.date(),
                        split.test_start.date(), split.test_end.date())
            train = panel_idx.iloc[split.train_idx].reset_index(drop=True)
            test = panel_idx.iloc[split.test_idx].reset_index(drop=True)

            # Drop rows where the primary target is missing (last 21d of feed)
            train = train.dropna(subset=["target_fwd_ret_21d"]).reset_index(drop=True)

            # Calibration slice (tail 90d of train)
            t_sorted = train.sort_values("date")
            calib_start = t_sorted["date"].max() - pd.Timedelta(days=90)
            calib = t_sorted[t_sorted["date"] >= calib_start].reset_index(drop=True)
            train_proper = t_sorted[t_sorted["date"] < calib_start].reset_index(drop=True)
            if len(calib) < 50 or len(train_proper) < 200:
                logger.warning("  fold %d: too few samples, skipping", split.fold)
                continue

            # ── HMM refit on train-only data (fix for leakage) ───────────
            # We pass the FULL price history but `end_date=train_end` causes
            # regimes.py to fit the HMM on pre-train_end data only and then
            # .predict() forward for test dates — no future leakage.
            prices_dt = prices.copy()
            prices_dt["date"] = pd.to_datetime(prices_dt["date"])
            _, fold_regimes = fit_regime_model(
                prices_dt,
                end_date=split.train_end,
                random_state=args.seed,
            )
            fold_regimes["date"] = pd.to_datetime(fold_regimes["date"])
            # Keep only test-window labels (each fold contributes its OOS labels)
            fold_regimes = fold_regimes[
                (fold_regimes["date"] >= split.test_start)
                & (fold_regimes["date"] < split.test_end)
            ].copy()
            fold_regimes["fold"] = split.fold
            regime_frames.append(fold_regimes)

            # ── Baseline ─────────────────────────────────────────────────
            base = BaselineForecaster(
                feature_cols=all_features,
                target_col="target_fwd_ret_21d",
                random_state=args.seed,
            ).fit(train_proper)
            # Conformal calibrate (calibration_df needs target_fwd_ret column)
            calib_for_cal = calib.copy()
            if "target_fwd_ret" not in calib_for_cal.columns:
                calib_for_cal["target_fwd_ret"] = calib_for_cal["target_fwd_ret_21d"]
            conf_base = SplitConformalWrapper(base_model=base, alpha=0.1).calibrate(
                calib_for_cal, target_col="target_fwd_ret_21d",
            )
            iv = conf_base.predict_interval(test)
            iv = iv.merge(
                test[["ticker", "date", "target_fwd_ret_21d"]],
                on=["ticker", "date"], how="inner",
            )
            iv["target_fwd_ret"] = iv["target_fwd_ret_21d"]
            iv["fold"] = split.fold
            iv["model"] = "baseline_gbr"
            baseline_preds.append(iv)

            # ── Transformer (multi-horizon + conformal) ──────────────────
            if have_torch:
                use_ticker_embed = args.ablation != "no_ticker_embed"
                use_causal_mask = args.ablation != "no_causal_mask"
                pooling = ("last" if args.ablation == "last_pooling"
                           else "attn")
                if args.ablation == "attn_pooling":
                    pooling = "attn"
                targets = (("target_fwd_ret_21d",)
                           if args.ablation == "no_multi_horizon"
                           else ("target_fwd_ret_5d", "target_fwd_ret_21d", "target_fwd_ret_63d"))
                # Loss: --rank-loss flag or --ablation pairwise_loss enables it
                loss_fn = "mse"
                if args.rank_loss or args.ablation == "pairwise_loss":
                    loss_fn = "pairwise"
                if args.ablation == "mse_loss":
                    loss_fn = "mse"
                cfg = TransformerConfig(
                    epochs=EPOCHS, batch_size=128, seq_len=SEQ_LEN,
                    use_ticker_embed=use_ticker_embed,
                    use_causal_mask=use_causal_mask,
                    pooling=pooling,
                    target_cols=targets,
                    primary_target="target_fwd_ret_21d",
                    loss_fn=loss_fn,
                    random_seed=args.seed,
                )
                tft = TransformerForecaster(feature_cols=all_features, config=cfg).fit(train_proper)
                conf_tft = SplitConformalWrapper(
                    base_model=tft, alpha=0.1, context_df=train_proper
                ).calibrate(calib)
                conf_tft.context_df = pd.concat([train_proper, calib], ignore_index=True)
                t_iv = conf_tft.predict_interval(test)
                if len(t_iv) > 0:
                    t_iv = t_iv.merge(
                        test[["ticker", "date", "target_fwd_ret_21d"]],
                        on=["ticker", "date"], how="inner"
                    )
                    t_iv = t_iv.rename(columns={"target_fwd_ret_21d": "target_fwd_ret"})
                    t_iv["target_fwd_ret_21d"] = t_iv["target_fwd_ret"]
                    t_iv["fold"] = split.fold
                    t_iv["model"] = "transformer"
                    transformer_preds.append(t_iv)

                # Export one attention sample per fold for visualisation
                try:
                    sample = tft.predict(
                        test.head(min(50, len(test))),
                        context_df=pd.concat([train_proper, calib], ignore_index=True),
                        return_attention=True,
                    )
                    if len(sample) > 0:
                        first = sample.iloc[0]
                        attention_samples.append({
                            "fold": split.fold,
                            "ticker": first["ticker"],
                            "date": str(pd.Timestamp(first["date"]).date()),
                            "attn": list(map(float, first["attn"])),
                        })
                except Exception as e:
                    logger.warning("attention export failed: %s", e)

        # ── Consolidate per-fold regime labels (drop duplicates, keep last) ─
        if regime_frames:
            regimes_all = pd.concat(regime_frames, ignore_index=True)
            regimes_all = regimes_all.drop_duplicates(subset=["date"], keep="last")
            regimes_all = regimes_all.sort_values("date").reset_index(drop=True)
            regimes_all.to_parquet(ART_DIR / "regime_labels.parquet", index=False)
            run.log_metric("n_regimes_distinct", int(regimes_all["regime"].nunique()))

        base_df = pd.concat(baseline_preds, ignore_index=True) if baseline_preds else pd.DataFrame()
        tft_df = pd.concat(transformer_preds, ignore_index=True) if transformer_preds else pd.DataFrame()
        if not base_df.empty:
            base_df.to_parquet(ART_DIR / "oos_preds_baseline.parquet", index=False)
        if not tft_df.empty:
            tft_df.to_parquet(ART_DIR / "oos_preds_transformer.parquet", index=False)

        if attention_samples:
            with open(ART_DIR / "attention_samples.json", "w", encoding="utf-8") as f:
                json.dump(attention_samples, f, indent=2)

        # ── 5. Regime-conditional eval + monthly rebalance ───────────────
        step("Step 5  Regime-conditional evaluation + monthly-rebalance backtest")
        from quant.ml.evaluation import evaluate_predictions, pretty_print, monthly_rebalance_backtest

        reg_for_eval = regimes_all if regime_frames else None
        summary = {}
        for name, preds in (("baseline_gbr", base_df), ("transformer", tft_df)):
            if preds.empty:
                continue
            rep = evaluate_predictions(preds, regimes_df=reg_for_eval, nominal_coverage=0.90)
            summary[name] = {
                "overall": rep.overall.to_dict(orient="records"),
                "per_regime": rep.per_regime.to_dict(orient="records"),
                "calibration": rep.calibration.to_dict(orient="records") if rep.calibration is not None else None,
            }
            print(f"\n🔹 {name}")
            print(pretty_print(rep))

            bt = monthly_rebalance_backtest(
                preds, top_k=3, target_col="target_fwd_ret_21d",
                horizon_days=21, tc_bps=args.tc_bps,
            )
            if "error" not in bt:
                summary[name]["monthly_rebalance"] = {
                    k: v for k, v in bt.items() if k != "pnl_df"
                }
                bt["pnl_df"].to_parquet(
                    ART_DIR / f"monthly_pnl_{name}.parquet", index=False
                )
                print(f"  Monthly rebal (tc={args.tc_bps:.0f}bps): "
                      f"sharpe_net={bt['sharpe']:+.2f}  sharpe_gross={bt['sharpe_gross']:+.2f}  "
                      f"total_net={bt['total_return']:+.2%}  maxDD={bt['max_drawdown']:+.2%}  "
                      f"turnover={bt['avg_turnover']:.2f}  n={bt['n_periods']}")
                run.log_metric(f"{name}_monthly_sharpe", bt["sharpe"])
                run.log_metric(f"{name}_monthly_sharpe_gross", bt["sharpe_gross"])
                run.log_metric(f"{name}_monthly_total_ret", bt["total_return"])
                run.log_metric(f"{name}_monthly_maxdd", bt["max_drawdown"])
                run.log_metric(f"{name}_avg_turnover", bt["avg_turnover"])
                run.log_metric(f"{name}_tc_drag", bt["total_tc_drag"])

            # KPIs
            ov = rep.overall.iloc[0].to_dict()
            for k in ("rmse", "dir_acc", "ic"):
                if k in ov:
                    run.log_metric(f"{name}_{k}", float(ov[k]))
            if rep.calibration is not None:
                ocov = rep.calibration[rep.calibration["regime"] == "overall"]
                if not ocov.empty:
                    run.log_metric(f"{name}_coverage", float(ocov["empirical_coverage"].iloc[0]))
                    run.log_metric(f"{name}_interval_width", float(ocov["mean_interval_width"].iloc[0]))

        with open(ART_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        run.log_artifact(str(ART_DIR / "evaluation_summary.json"))

        # ── 6. Uncertainty-aware + sector-neutral allocator ──────────────
        step("Step 6  Allocators: vanilla / width-scaled / sector-neutral")
        from quant.ml.allocator import run_allocator
        alloc_summary = {}
        policies = ("vanilla", "width_scaled", "sector_neutral")
        for name, preds in (("baseline_gbr", base_df), ("transformer", tft_df)):
            if preds.empty:
                continue
            results_by_policy = {}
            for pol in policies:
                r = run_allocator(
                    preds, policy=pol, regimes_df=reg_for_eval,
                    tc_bps=args.tc_bps, horizon_days=21,
                )
                if "error" in r:
                    continue
                results_by_policy[pol] = r
            print(f"\n{name}:")
            for pol, r in results_by_policy.items():
                print(f"  {pol:15s}  sharpe_net={r['sharpe']:+.2f}  "
                      f"sharpe_gross={r['sharpe_gross']:+.2f}  "
                      f"total={r['total_return']:+.2%}  maxDD={r['max_drawdown']:+.2%}  "
                      f"gross={r['gross_leverage_mean']:.2f}  "
                      f"turnover={r['avg_turnover']:.2f}")
                run.log_metric(f"{name}_{pol}_sharpe", r["sharpe"])
                run.log_metric(f"{name}_{pol}_sharpe_gross", r["sharpe_gross"])
                run.log_metric(f"{name}_{pol}_maxdd", r["max_drawdown"])
                run.log_metric(f"{name}_{pol}_turnover", r["avg_turnover"])
            if results_by_policy:
                alloc_summary[name] = {}
                for pol, r in results_by_policy.items():
                    alloc_summary[name][pol] = {
                        k: v for k, v in r.items() if k not in ("weights_df", "pnl_df", "per_regime")
                    }
                    if "per_regime" in r and isinstance(r["per_regime"], pd.DataFrame):
                        alloc_summary[name][f"{pol}_per_regime"] = r["per_regime"].to_dict(orient="records")
                    r["pnl_df"].to_parquet(
                        ART_DIR / f"alloc_pnl_{name}_{pol}.parquet", index=False
                    )

        with open(ART_DIR / "allocator_summary.json", "w", encoding="utf-8") as f:
            json.dump(alloc_summary, f, indent=2, default=str)

        # ── 6b. Statistical significance: DM + bootstrap + PBO ───────────
        step("Step 6b  Statistical significance (Diebold-Mariano, bootstrap CI, PBO)")
        from quant.ml.statistics import diebold_mariano, bootstrap_sharpe_ci, pbo
        sig = {}
        # DM test on squared errors of baseline vs transformer
        if (not base_df.empty) and (not tft_df.empty):
            merged = base_df[["ticker", "date", "prediction", "target_fwd_ret"]].merge(
                tft_df[["ticker", "date", "prediction"]],
                on=["ticker", "date"], how="inner", suffixes=("_base", "_tft"),
            )
            if len(merged) >= 30:
                se_base = (merged["target_fwd_ret"] - merged["prediction_base"]) ** 2
                se_tft = (merged["target_fwd_ret"] - merged["prediction_tft"]) ** 2
                dm = diebold_mariano(se_tft.values, se_base.values, horizon=21)
                print(f"  DM (Transformer vs Baseline, squared error, h=21):")
                print(f"    stat={dm.dm_stat:+.3f}  p={dm.p_value:.4f}  → better={dm.better_model}")
                sig["diebold_mariano"] = {
                    "mean_diff": dm.mean_diff, "dm_stat": dm.dm_stat,
                    "p_value": dm.p_value, "n": dm.n,
                    "better": "transformer" if dm.better_model == "A"
                              else ("baseline" if dm.better_model == "B" else "tie"),
                }
                run.log_metric("dm_stat", dm.dm_stat)
                run.log_metric("dm_p_value", dm.p_value)

        # Bootstrap Sharpe CI on the net monthly PnL for each model
        for name in ("baseline_gbr", "transformer"):
            p_path = ART_DIR / f"monthly_pnl_{name}.parquet"
            if p_path.exists():
                pnl = pd.read_parquet(p_path)
                if len(pnl) >= 6:
                    ci = bootstrap_sharpe_ci(
                        pnl["ls_return"].values, periods_per_year=12,
                        n_boot=2000, seed=args.seed,
                    )
                    print(f"  {name} monthly Sharpe (net) = {ci['point']:+.2f}  "
                          f"95% CI [{ci['lo']:+.2f}, {ci['hi']:+.2f}]  (n={ci['n']})")
                    sig[f"{name}_sharpe_ci"] = ci
                    run.log_metric(f"{name}_sharpe_ci_lo", ci["lo"])
                    run.log_metric(f"{name}_sharpe_ci_hi", ci["hi"])

        # PBO across allocator policies × models — treats (model, policy) pairs as
        # candidate strategies and checks whether the best-IS is the best OOS.
        pnl_matrix_rows = []
        strat_names = []
        for name in ("baseline_gbr", "transformer"):
            for pol in policies:
                p = ART_DIR / f"alloc_pnl_{name}_{pol}.parquet"
                if p.exists():
                    df_p = pd.read_parquet(p).sort_values("date")
                    pnl_matrix_rows.append(df_p["ret"].values)
                    strat_names.append(f"{name}:{pol}")
        if len(pnl_matrix_rows) >= 2:
            min_len = min(len(x) for x in pnl_matrix_rows)
            if min_len >= 8:
                M = np.column_stack([x[:min_len] for x in pnl_matrix_rows])
                pbo_res = pbo(M)
                print(f"  PBO across {len(strat_names)} (model, policy) pairs: "
                      f"{pbo_res['pbo']:.2f}  "
                      f"(trials={pbo_res['n_trials']})")
                sig["pbo"] = {**pbo_res, "strategies": strat_names}
                run.log_metric("pbo", pbo_res["pbo"])

        with open(ART_DIR / "significance.json", "w", encoding="utf-8") as f:
            json.dump(sig, f, indent=2, default=str)
        run.log_artifact(str(ART_DIR / "significance.json"))

        # ── 6c. Feature drift detection ──────────────────────────────────
        step("Step 6c  Feature drift detection (reference=train / production=latest OOS)")
        try:
            from quant.ml.drift import detect_feature_drift, drift_summary_line
            # Reference: first 60% of panel. Production: last 20%.
            cut60 = panel["date"].quantile(0.6)
            cut80 = panel["date"].quantile(0.8)
            drift_report = detect_feature_drift(
                panel, all_features,
                reference_start=panel["date"].min().strftime("%Y-%m-%d"),
                reference_end=cut60.strftime("%Y-%m-%d"),
                production_start=cut80.strftime("%Y-%m-%d"),
                production_end=panel["date"].max().strftime("%Y-%m-%d"),
            )
            print(" ", drift_summary_line(drift_report))
            drift_report.per_feature.to_csv(ART_DIR / "drift_report.csv", index=False)
            run.log_metric("n_drifted_features", drift_report.n_drifted)
            # Top-5 drifted features
            flagged = drift_report.per_feature[drift_report.per_feature["drift"]].head(5)
            if not flagged.empty:
                print("  Top drifted:")
                for _, row in flagged.iterrows():
                    print(f"    {row['feature']:30s}  ks={row['ks_stat']:.3f}  p={row['p_value']:.4f}")
        except Exception as e:
            logger.warning("drift detection failed: %s", e)

        # ── 7. SHAP ──────────────────────────────────────────────────────
        step("Step 7  SHAP attribution on baseline")
        from quant.ml.shap_explain import explain_baseline
        panel_21 = panel.dropna(subset=["target_fwd_ret_21d"]).copy()
        final_base = BaselineForecaster(
            feature_cols=all_features, target_col="target_fwd_ret_21d",
        ).fit(panel_21)
        shap_report = explain_baseline(
            final_base,
            panel_21.sample(min(2000, len(panel_21)), random_state=args.seed),
            regimes_df=reg_for_eval,
            max_samples=2000,
        )
        shap_report.global_importance.to_csv(ART_DIR / "shap_global.csv", index=False)
        if shap_report.per_regime is not None:
            shap_report.per_regime.to_csv(ART_DIR / "shap_per_regime.csv", index=False)
        print("  top-10 features:")
        print(shap_report.global_importance.head(10).to_string(index=False))
        run.log_artifact(str(ART_DIR / "shap_global.csv"))

        # ── 8. Plots ─────────────────────────────────────────────────────
        step("Step 8  Rendering plots")
        try:
            _render_plots(summary, base_df, tft_df, reg_for_eval, attention_samples)
            print(f"  wrote PNGs to {ART_DIR}/")
            for f in ("plot_long_short_pnl.png", "plot_conformal_coverage.png",
                      "plot_dir_acc_by_regime.png", "plot_attention_weights.png",
                      "plot_allocator_compare.png"):
                p = ART_DIR / f
                if p.exists():
                    run.log_artifact(str(p))
        except Exception as e:
            logger.warning("plot rendering failed: %s", e)

        # ── 9. Postgres (best-effort) ────────────────────────────────────
        step("Step 9  Loading predictions to PostgreSQL (best-effort)")
        _load_to_postgres(base_df, tft_df)

    banner("✅  ML PIPELINE COMPLETE")
    print(f"Artifacts: {ART_DIR.resolve()}")
    dt = (datetime.now() - t_start).total_seconds()
    print(f"Total runtime: {dt:.1f}s  (fast={args.fast})")


# ─────────────────────────────────────────────────────────────────────────────
def _render_plots(reports: dict, base_df: pd.DataFrame, tft_df: pd.DataFrame,
                  regimes, attention_samples: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Per-regime directional accuracy
    if reports:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xs, ys, colors = [], [], []
        palette = {"baseline_gbr": "#ea4335", "transformer": "#1a73e8"}
        for model_name, rep in reports.items():
            for r in rep["per_regime"]:
                xs.append(f"{r['regime']}\n({model_name[:10]})")
                ys.append(float(r["dir_acc"]))
                colors.append(palette.get(model_name, "grey"))
        if xs:
            ax.bar(xs, ys, color=colors, alpha=0.85)
            ax.axhline(0.5, color="grey", lw=1, ls="--", label="coin flip")
            ax.set_ylabel("Directional accuracy")
            ax.set_title("Per-regime OOS directional accuracy")
            ax.legend()
            plt.tight_layout()
            plt.savefig(ART_DIR / "plot_dir_acc_by_regime.png", dpi=140)
        plt.close(fig)

    # 2. Conformal coverage
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs, ys, cs = [], [], []
    palette = {"baseline_gbr": "#ea4335", "transformer": "#1a73e8"}
    for model_name, rep in reports.items():
        if not rep.get("calibration"):
            continue
        for r in rep["calibration"]:
            xs.append(f"{r['regime']}\n({model_name[:10]})")
            ys.append(float(r["empirical_coverage"]))
            cs.append(palette.get(model_name, "grey"))
    if ys:
        ax.bar(xs, ys, alpha=0.85, color=cs)
        ax.axhline(0.90, color="red", lw=1.3, ls="--", label="nominal 0.90")
        ax.set_ylabel("Empirical coverage")
        ax.set_title("Conformal calibration: empirical vs nominal coverage")
        ax.set_ylim(0, 1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig(ART_DIR / "plot_conformal_coverage.png", dpi=140)
    plt.close(fig)

    # 3. Monthly-rebalance PnL comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in [("baseline_gbr", "#ea4335"), ("transformer", "#1a73e8")]:
        p = ART_DIR / f"monthly_pnl_{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["cum"] = (1 + df["ls_return"]).cumprod()
            ax.plot(pd.to_datetime(df["date"]), df["cum"], label=f"{name} (monthly rebal)",
                    color=color, lw=2)
    ax.set_title("Monthly-rebalance long-short (top-3 − bottom-3) — honest cumulative return")
    ax.set_ylabel("Cumulative return (1=flat)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ART_DIR / "plot_long_short_pnl.png", dpi=140)
    plt.close(fig)

    # 4. Allocator comparison (vanilla / width-scaled / sector-neutral)
    fig, ax = plt.subplots(figsize=(10, 5))
    drew = False
    policy_palette = {
        "vanilla": "grey",
        "width_scaled": "#1a73e8",
        "sector_neutral": "#34a853",
    }
    for policy, color in policy_palette.items():
        p = ART_DIR / f"alloc_pnl_transformer_{policy}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["cum"] = (1 + df["ret"]).cumprod()
            ax.plot(pd.to_datetime(df["date"]), df["cum"],
                    label=f"transformer • {policy}", color=color, lw=2)
            drew = True
    if drew:
        ax.set_title("Allocator comparison (transformer signal, net of tc)")
        ax.set_ylabel("Cumulative return (1=flat)")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.savefig(ART_DIR / "plot_allocator_compare.png", dpi=140)
    plt.close(fig)

    # 5. Attention weights — plot first sample
    if attention_samples:
        sample = attention_samples[0]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(range(len(sample["attn"])), sample["attn"], color="#1a73e8")
        ax.set_xlabel("Timesteps back (0 = most recent)")
        ax.set_ylabel("Attention weight")
        ax.set_title(
            f"Transformer attention pattern — {sample['ticker']} on {sample['date']} (fold {sample['fold']})"
        )
        plt.tight_layout()
        plt.savefig(ART_DIR / "plot_attention_weights.png", dpi=140)
        plt.close(fig)


def _load_to_postgres(base_df: pd.DataFrame, tft_df: pd.DataFrame) -> None:
    try:
        import psycopg2
        from pipeline.load import get_connection
    except Exception as e:
        logger.info("  psycopg2 unavailable (%s), skipping", e)
        return

    try:
        conn = get_connection()
    except Exception as e:
        logger.info("  Postgres unreachable (%s), skipping", e)
        return

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id SERIAL PRIMARY KEY,
            model VARCHAR(40),
            ticker VARCHAR(20),
            date DATE,
            prediction FLOAT,
            lower_bound FLOAT,
            upper_bound FLOAT,
            target_realised FLOAT,
            fold INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (model, ticker, date, fold)
        )
    """)
    conn.commit()

    import psycopg2.extras as pe
    for df, tag in ((base_df, "baseline_gbr"), (tft_df, "transformer")):
        if df is None or df.empty:
            continue
        sql = """
            INSERT INTO ml_predictions
              (model, ticker, date, prediction, lower_bound, upper_bound, target_realised, fold)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model, ticker, date, fold) DO NOTHING
        """
        rows = [
            (tag, r["ticker"], pd.Timestamp(r["date"]).date(),
             float(r["prediction"]), float(r["lower"]), float(r["upper"]),
             (float(r["target_fwd_ret"]) if pd.notna(r.get("target_fwd_ret", np.nan)) else None),
             int(r.get("fold", -1)))
            for _, r in df.iterrows()
        ]
        pe.execute_batch(cur, sql, rows, page_size=500)
        conn.commit()
        logger.info("  inserted %d rows for model=%s", len(rows), tag)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
