"""
service/app.py — FastAPI serving endpoint for the ML forecaster.

Endpoints
---------
GET  /health
GET  /model/info
GET  /predictions/latest?model=transformer&top_k=5
GET  /predictions/{model}/{ticker}?start=YYYY-MM-DD&end=YYYY-MM-DD
POST /allocate  →  body: { model, policy, top_k }  →  latest allocator weights
GET  /regimes/latest

Prediction data is served from the `artifacts/oos_preds_*.parquet` files
produced by `run_ml_pipeline.py`. The service is read-only — it exposes the
*already-computed* OOS predictions rather than retraining on request. This
mirrors the standard MLOps split: an offline training job writes a model
artifact; an online service loads and serves it.

Run locally:
    uvicorn service.app:app --reload --port 8088
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ART_DIR = ROOT / "artifacts"

app = FastAPI(
    title="Equity Intelligence ML API",
    version="2.0",
    description=(
        "Read-only service exposing out-of-sample ML predictions, "
        "conformal prediction intervals, regime labels, and uncertainty-aware "
        "portfolio weights produced by the offline `run_ml_pipeline.py` job."
    ),
)


# ── Lazy, cached loaders ────────────────────────────────────────────────────
_cache: dict = {}


def _load_parquet(name: str) -> Optional[pd.DataFrame]:
    p = ART_DIR / name
    if not p.exists():
        return None
    if name not in _cache:
        df = pd.read_parquet(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        _cache[name] = df
    return _cache[name]


def _load_json(name: str) -> Optional[dict]:
    p = ART_DIR / name
    if not p.exists():
        return None
    if name not in _cache:
        _cache[name] = json.loads(p.read_text(encoding="utf-8"))
    return _cache[name]


def _clear_cache():
    _cache.clear()


# ── Response models ─────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    artifacts_dir: str
    available_models: List[str]
    generated_at: str


class PredictionRow(BaseModel):
    ticker: str
    date: str
    prediction: float
    lower: float
    upper: float
    target_realised: Optional[float] = None
    regime: Optional[str] = None


class AllocationRequest(BaseModel):
    model: Literal["transformer", "baseline_gbr"] = "transformer"
    policy: Literal["vanilla", "width_scaled"] = "width_scaled"
    top_k: int = Field(3, ge=1, le=20)


class AllocationRow(BaseModel):
    ticker: str
    weight: float
    prediction: float


class AllocationResponse(BaseModel):
    model: str
    policy: str
    rebalance_date: str
    gross_leverage: float
    weights: List[AllocationRow]


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    available = []
    for fname, tag in (
        ("oos_preds_baseline.parquet", "baseline_gbr"),
        ("oos_preds_transformer.parquet", "transformer"),
    ):
        if (ART_DIR / fname).exists():
            available.append(tag)
    return {
        "status": "ok" if available else "no_predictions",
        "artifacts_dir": str(ART_DIR.resolve()),
        "available_models": available,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/model/info")
def model_info():
    summ = _load_json("evaluation_summary.json") or {}
    alloc = _load_json("allocator_summary.json") or {}
    info = {}
    for m in ("baseline_gbr", "transformer"):
        if m in summ:
            ov = summ[m]["overall"][0] if summ[m].get("overall") else {}
            cov = None
            if summ[m].get("calibration"):
                overall = [c for c in summ[m]["calibration"] if c["regime"] == "overall"]
                if overall:
                    cov = overall[0]
            info[m] = {
                "oos_rows": int(ov.get("n_samples", 0)),
                "directional_accuracy": round(float(ov.get("dir_acc", 0)), 4),
                "information_coefficient": round(float(ov.get("ic", 0)), 4),
                "rmse": round(float(ov.get("rmse", 0)), 6),
                "monthly_rebalance": summ[m].get("monthly_rebalance"),
                "conformal_coverage_overall": cov,
                "allocator": alloc.get(m),
            }
    return info


@app.get("/predictions/latest")
def predictions_latest(
    model: Literal["transformer", "baseline_gbr"] = "transformer",
    top_k: int = Query(5, ge=1, le=50),
):
    df = _load_parquet(f"oos_preds_{model}.parquet")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"no predictions for {model}")
    latest_date = df["date"].max()
    day = df[df["date"] == latest_date].copy()
    day = day.sort_values("prediction", ascending=False).head(top_k)
    return {
        "model": model,
        "date": latest_date.strftime("%Y-%m-%d"),
        "top_k": top_k,
        "rows": [
            {
                "ticker": r["ticker"],
                "prediction": float(r["prediction"]),
                "lower": float(r["lower"]),
                "upper": float(r["upper"]),
                "interval_width": float(r["upper"] - r["lower"]),
                "target_realised": (
                    float(r["target_fwd_ret"]) if pd.notna(r.get("target_fwd_ret")) else None
                ),
            }
            for _, r in day.iterrows()
        ],
    }


@app.get("/predictions/{model}/{ticker}", response_model=List[PredictionRow])
def predictions_for_ticker(
    model: Literal["transformer", "baseline_gbr"],
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    df = _load_parquet(f"oos_preds_{model}.parquet")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"no predictions for {model}")
    sub = df[df["ticker"] == ticker].copy()
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"no predictions for ticker={ticker}")
    if start:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end:
        sub = sub[sub["date"] <= pd.Timestamp(end)]
    sub = sub.sort_values("date")

    regimes = _load_parquet("regime_labels.parquet")
    if regimes is not None:
        sub = sub.merge(regimes[["date", "regime"]], on="date", how="left")

    return [
        PredictionRow(
            ticker=r["ticker"],
            date=r["date"].strftime("%Y-%m-%d"),
            prediction=float(r["prediction"]),
            lower=float(r["lower"]),
            upper=float(r["upper"]),
            target_realised=(float(r["target_fwd_ret"])
                             if pd.notna(r.get("target_fwd_ret")) else None),
            regime=(str(r["regime"]) if "regime" in r and pd.notna(r.get("regime")) else None),
        )
        for _, r in sub.iterrows()
    ]


@app.post("/allocate", response_model=AllocationResponse)
def allocate(req: AllocationRequest):
    df = _load_parquet(f"oos_preds_{req.model}.parquet")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"no predictions for {req.model}")

    from quant.ml.allocator import vanilla_ls_weights, width_scaled_weights

    latest_date = df["date"].max()
    day = df[df["date"] == latest_date].copy()
    if len(day) < 2 * req.top_k:
        raise HTTPException(
            status_code=400,
            detail=f"not enough rows for top_k={req.top_k} (have {len(day)})",
        )

    fn = width_scaled_weights if req.policy == "width_scaled" else vanilla_ls_weights
    w = fn(day, top_k=req.top_k)
    gross = float(w["weight"].abs().sum())

    return AllocationResponse(
        model=req.model,
        policy=req.policy,
        rebalance_date=latest_date.strftime("%Y-%m-%d"),
        gross_leverage=gross,
        weights=[
            AllocationRow(
                ticker=row["ticker"],
                weight=float(row["weight"]),
                prediction=float(row["prediction"]),
            )
            for _, row in w.sort_values("weight", ascending=False).iterrows()
        ],
    )


@app.get("/regimes/latest")
def regimes_latest(n: int = Query(30, ge=1, le=500)):
    df = _load_parquet("regime_labels.parquet")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="no regime labels available")
    tail = df.sort_values("date").tail(n)
    return {
        "rows": [
            {"date": r["date"].strftime("%Y-%m-%d"), "regime": r["regime"]}
            for _, r in tail.iterrows()
        ]
    }


@app.post("/admin/reload")
def admin_reload():
    """Clear in-process artifact cache — useful after a pipeline re-run."""
    _clear_cache()
    return {"status": "reloaded"}
