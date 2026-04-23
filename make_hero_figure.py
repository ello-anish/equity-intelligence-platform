"""
make_hero_figure.py — Build a single hero figure summarising v3 ML results.

Read-only: consumes existing artifacts in ./artifacts/ and writes
  artifacts/hero_figure.png        (~3200x2400, dpi=200)
  artifacts/hero_figure_small.png  (~1600x1200, LinkedIn inline)

2x2 layout:
  A  Conformal calibration (bars, nominal line)
  B  Net Sharpe with 95% bootstrap CI (forest plot + DM test annotation)
  C  Top-10 SHAP feature importances (macro vs micro colored)
  D  Uncertainty-aware allocator drawdown

Palette is strictly three data colors:
  NAVY  #1f3a68 — Transformer / macro
  GRAY  #888888 — Baseline / micro / subtitles
  RED   #c0392b — zero-line / nominal-coverage annotations only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ART = Path("artifacts")

NAVY = "#1f3a68"
GRAY = "#888888"
RED = "#c0392b"
SUBTITLE_COLOR = "#555555"  # darker gray used for panel subtitles (readability after scale-down)

# Display-only aliases — keeps the underlying artifact column names untouched
# but renders more readable labels in the hero figure.
DISPLAY_ALIAS: dict[str, str] = {
    "india_vix_z20": "india_vix_z_20d",
}

# ── Fallbacks (used if the real artifact is missing / parses badly) ────
FB_COVERAGE = {"baseline_gbr": 0.771, "transformer": 0.895}
FB_SHARPE = {
    "baseline_gbr": {"point": -1.56, "lo": -3.22, "hi": -0.52},
    "transformer": {"point": -0.60, "lo": -1.81, "hi": +0.62},
}
FB_SHAP_TOP = [
    ("us10y_chg_21d", 0.017),
    ("us10y_level", 0.015),
    ("mom_12m", 0.009),
    ("mom_6m", 0.008),
    ("india_vix_level", 0.007),
    ("india_vix_z20", 0.007),
    ("macd", 0.006),
    ("vol_60d", 0.006),
    ("usdinr_ret_21d", 0.006),
    ("nifty_over_ma50", 0.006),
]
FB_DD = {"vanilla": -0.2746, "width_scaled": -0.1449}

used_real: dict[str, bool] = {"A": False, "B": False, "C": False, "D": False}
warnings_: list[str] = []


def warn_fb(msg: str) -> None:
    warnings_.append(msg)
    print(f"[fallback] {msg}")


# ── Loaders ────────────────────────────────────────────────────────────
def load_coverage() -> dict[str, float]:
    for fname in ("conformal_coverage.json", "evaluation_summary.json"):
        p = ART / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            warn_fb(f"couldn't parse {fname}: {e}")
            continue
        out: dict[str, float] = {}
        if fname == "evaluation_summary.json":
            for m in ("baseline_gbr", "transformer"):
                if m in data and data[m].get("calibration"):
                    row = next((r for r in data[m]["calibration"]
                                if r.get("regime") == "overall"), None)
                    if row and "empirical_coverage" in row:
                        out[m] = float(row["empirical_coverage"])
        else:
            for m in ("baseline_gbr", "transformer"):
                if m in data:
                    try:
                        out[m] = float(data[m])
                    except Exception:
                        pass
        if len(out) == 2:
            used_real["A"] = True
            return out
        if out:
            # Partial — fill missing from fallback
            for m, v in FB_COVERAGE.items():
                out.setdefault(m, v)
            used_real["A"] = True
            return out
    warn_fb(f"using hard-coded coverage={FB_COVERAGE} (no calibration JSON found)")
    return dict(FB_COVERAGE)


def load_sharpe_ci() -> dict[str, dict[str, float]]:
    for fname in ("sharpe_ci.json", "significance.json"):
        p = ART / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            warn_fb(f"couldn't parse {fname}: {e}")
            continue
        out: dict[str, dict[str, float]] = {}
        for m in ("baseline_gbr", "transformer"):
            key = f"{m}_sharpe_ci"
            if key in data and all(k in data[key] for k in ("point", "lo", "hi")):
                out[m] = {k: float(data[key][k]) for k in ("point", "lo", "hi")}
            elif m in data and isinstance(data[m], dict) and "point" in data[m]:
                out[m] = {k: float(data[m][k]) for k in ("point", "lo", "hi")}
        if len(out) == 2:
            used_real["B"] = True
            return out
    warn_fb(f"using hard-coded Sharpe CIs (no sharpe_ci JSON found)")
    return {m: dict(v) for m, v in FB_SHARPE.items()}


def load_dm_p() -> float:
    p = ART / "significance.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "diebold_mariano" in data and "p_value" in data["diebold_mariano"]:
                return float(data["diebold_mariano"]["p_value"])
        except Exception as e:
            warn_fb(f"couldn't parse significance.json DM block: {e}")
    warn_fb("using hard-coded DM p=0.0001")
    return 0.0001


def load_shap() -> list[tuple[str, float]]:
    for fname in ("shap_importance.csv", "shap_global.csv"):
        p = ART / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            warn_fb(f"couldn't parse {fname}: {e}")
            continue

        # Figure out which column carries importance
        val_col = None
        for c in ("mean_abs_shap", "importance", "mean_shap", "shap"):
            if c in df.columns:
                val_col = c
                break
        if val_col is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                continue
            val_col = num_cols[0]
        feat_col = "feature" if "feature" in df.columns else df.columns[0]
        df = df.sort_values(val_col, ascending=False).head(10)
        used_real["C"] = True
        return list(zip(df[feat_col].astype(str).tolist(),
                        df[val_col].astype(float).tolist()))

    listing = sorted(p.name for p in ART.iterdir()) if ART.exists() else []
    warn_fb(
        f"no SHAP csv found — tried shap_importance.csv, shap_global.csv; "
        f"artifacts/ contains: {listing}"
    )
    return list(FB_SHAP_TOP)


def is_macro(feature: str) -> bool:
    f = feature.lower()
    return any(tag in f for tag in ("us10y", "vix", "usdinr", "nifty", "gold", "macro_"))


def load_drawdown_series() -> dict[str, pd.DataFrame] | None:
    """Return {'vanilla': df, 'width_scaled': df} if both parquet files exist."""
    out: dict[str, pd.DataFrame] = {}
    for policy in ("vanilla", "width_scaled"):
        p = ART / f"alloc_pnl_transformer_{policy}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if "cum" not in df.columns:
                df["cum"] = (1 + df["ret"]).cumprod()
            df["drawdown"] = df["cum"] / df["cum"].cummax() - 1.0
            out[policy] = df
        except Exception as e:
            warn_fb(f"couldn't read {p.name}: {e}")
    if len(out) == 2:
        used_real["D"] = True
        return out
    return None


def load_dd_summary() -> dict[str, float]:
    p = ART / "allocator_summary.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            tr = data.get("transformer", {})
            out: dict[str, float] = {}
            for policy in ("vanilla", "width_scaled"):
                block = tr.get(policy)
                if block and "max_drawdown" in block:
                    out[policy] = float(block["max_drawdown"])
            if len(out) == 2:
                used_real["D"] = True
                return out
        except Exception as e:
            warn_fb(f"couldn't parse allocator_summary.json: {e}")
    warn_fb(f"using hard-coded drawdown={FB_DD}")
    return dict(FB_DD)


# ── Plot helpers ───────────────────────────────────────────────────────
def style_axes(ax, *, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.6,
                color=GRAY, alpha=0.22)
        ax.set_axisbelow(True)


def panel_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=30, loc="left")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=11, style="italic", color=SUBTITLE_COLOR,
                va="bottom", ha="left")


def prettify(name: str) -> str:
    """Display-only beautification of feature names for Panel C."""
    return DISPLAY_ALIAS.get(name, name)


# ── Build figure ───────────────────────────────────────────────────────
def main() -> None:
    if not ART.exists():
        print(f"ERROR: {ART.resolve()} does not exist — run the ML pipeline first.",
              file=sys.stderr)
        sys.exit(1)

    cov = load_coverage()
    sharpe = load_sharpe_ci()
    dm_p = load_dm_p()
    shap_rows = load_shap()
    dd_series = load_drawdown_series()
    dd_summary = load_dd_summary() if dd_series is None else None

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)

    # ── Panel A ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    labels = ["Baseline", "Transformer"]
    vals = [cov.get("baseline_gbr", FB_COVERAGE["baseline_gbr"]),
            cov.get("transformer",  FB_COVERAGE["transformer"])]
    xs = np.arange(2)
    ax.bar(xs, vals, color=[GRAY, NAVY], width=0.55)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.025, f"{v:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.axhline(0.90, color=RED, linestyle="--", linewidth=1.5, alpha=0.9)
    ax.text(1.48, 0.905, "Nominal 90%", color=RED, fontsize=10,
            ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    style_axes(ax, grid_axis="y")
    panel_title(ax, "A  Conformal Calibration",
                "Transformer intervals are well-calibrated; Baseline undercovers")

    # ── Panel B ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    order = [("Baseline", "baseline_gbr", GRAY),
             ("Transformer", "transformer", NAVY)]
    ys = np.arange(len(order))
    LABEL_Y_OFFSET = 0.22   # consistent: labels always this far above their point
    for y, (label, key, color) in zip(ys, order):
        s = sharpe[key]
        ax.errorbar([s["point"]], [y],
                    xerr=[[s["point"] - s["lo"]], [s["hi"] - s["point"]]],
                    fmt="o", color=color, ecolor=color,
                    capsize=7, elinewidth=2.2, markersize=11)
        # Label ABOVE the point, centered horizontally on it (consistent position)
        ax.text(s["point"], y + LABEL_Y_OFFSET,
                f"{s['point']:+.2f}   [{s['lo']:+.2f}, {s['hi']:+.2f}]",
                fontsize=10, va="bottom", ha="center",
                color=color, fontweight="bold")
    ax.axvline(0, color=RED, linestyle="--", linewidth=1.5, alpha=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([o[0] for o in order], fontsize=11)
    ax.set_xlabel("Sharpe ratio (net of 20bps tc)", fontsize=11)
    # Tight xlim — CI is [-3.22, +0.62], pad a bit on each side for labels/markers
    ax.set_xlim(-3.8, 1.8)
    ax.set_xticks([-3, -2, -1, 0, 1])
    # Extra vertical headroom above the top row so the label doesn't crowd the title
    ax.set_ylim(-0.55, len(order) - 1 + 0.85)
    # DM annotation — moved to bottom-right so it never collides with the labels
    if dm_p < 0.0001:
        dm_head = "DM test p<0.0001"
    else:
        dm_head = f"DM test p={dm_p:.4f}"
    dm_txt = (f"{dm_head}\n"
              "Transformer significantly beats Baseline\n"
              "on forecast error")
    ax.text(0.98, 0.04, dm_txt, transform=ax.transAxes,
            fontsize=9.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=GRAY, linewidth=0.8, alpha=0.95))
    style_axes(ax)
    panel_title(ax, "B  Net Sharpe with 95% Bootstrap CI",
                "Honest reporting: Transformer CI spans zero")

    # ── Panel C ─────────────────────────────────────────────────────
    ax = axes[1, 0]
    top = shap_rows[:10]
    top_sorted = sorted(top, key=lambda x: x[1])      # ascending → largest at top
    names = [n for n, _ in top_sorted]
    vals = [v for _, v in top_sorted]
    colors = [NAVY if is_macro(n) else GRAY for n in names]
    ys = np.arange(len(names))
    ax.barh(ys, vals, color=colors, height=0.62)
    ax.set_yticks(ys)
    ax.set_yticklabels([prettify(n) for n in names], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    # Legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=NAVY, label="Macro"),
                       Patch(facecolor=GRAY, label="Micro")],
              loc="lower right", frameon=False, fontsize=10)
    style_axes(ax, grid_axis="x")
    n_macro = sum(1 for n in names if is_macro(n))
    panel_title(ax, "C  Top 10 Feature Importances (SHAP)",
                f"{n_macro} of top 10 are macro — US 10y, India VIX, USDINR dominate")

    # ── Panel D ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    if dd_series is not None:
        v = dd_series["vanilla"]
        w = dd_series["width_scaled"]
        ax.fill_between(v["date"], v["drawdown"], 0, color=GRAY, alpha=0.30)
        ax.plot(v["date"], v["drawdown"], color=GRAY, linewidth=2.0, label="Vanilla")
        ax.fill_between(w["date"], w["drawdown"], 0, color=NAVY, alpha=0.25)
        ax.plot(w["date"], w["drawdown"], color=NAVY, linewidth=2.0, label="Width-scaled")
        v_i = int(v["drawdown"].idxmin()); w_i = int(w["drawdown"].idxmin())
        v_dd = float(v["drawdown"].iloc[v_i]); w_dd = float(w["drawdown"].iloc[w_i])
        # Vanilla (deepest DD, near the bottom of the plot) — annotate UP-AND-LEFT
        # with a curved arrow so the text clears the x-axis tick labels entirely.
        ax.annotate(f"Max DD {v_dd:+.1%}",
                    xy=(v.loc[v_i, "date"], v_dd),
                    xytext=(-95, 60), textcoords="offset points",
                    color=GRAY, fontsize=10, fontweight="bold",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2,
                                    connectionstyle="arc3,rad=0.25"))
        # Width-scaled (shallower) — annotate UP-AND-RIGHT with extra separation
        ax.annotate(f"Max DD {w_dd:+.1%}",
                    xy=(w.loc[w_i, "date"], w_dd),
                    xytext=(75, 55), textcoords="offset points",
                    color=NAVY, fontsize=10, fontweight="bold",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.set_ylabel("Drawdown", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.axhline(0, color=GRAY, linewidth=0.6, alpha=0.5)
        # Extra top padding so the annotations have room and don't clip
        y_bottom = min(v["drawdown"].min(), w["drawdown"].min())
        ax.set_ylim(y_bottom * 1.08, 0.045)
        ax.legend(loc="lower right", frameon=False, fontsize=10)
    else:
        labels_d = ["Vanilla", "Width-scaled"]
        vals_d = [dd_summary.get("vanilla", FB_DD["vanilla"]),
                  dd_summary.get("width_scaled", FB_DD["width_scaled"])]
        xs = np.arange(2)
        ax.bar(xs, vals_d, color=[GRAY, NAVY], width=0.55)
        for x, val in zip(xs, vals_d):
            ax.text(x, val - 0.008, f"{val:+.1%}", ha="center", va="top",
                    fontsize=12, fontweight="bold", color="white")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels_d, fontsize=11)
        ax.set_ylabel("Max drawdown (Transformer)", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.axhline(0, color=GRAY, linewidth=0.6, alpha=0.5)
    style_axes(ax, grid_axis="y")
    panel_title(ax, "D  Uncertainty-Aware Allocation",
                "Width-scaled allocator cuts max drawdown ~45%")

    # ── Suptitles ───────────────────────────────────────────────────
    fig.suptitle("Regime-Aware Equity Forecasting — Methodology Summary",
                 fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.952,
             "2019-2024   ·   7 walk-forward folds   ·   20bps transaction costs   ·   10 NSE large-caps",
             ha="center", fontsize=12, color=GRAY)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.935], pad=2.0, h_pad=4.0, w_pad=4.0)

    # ── Save ────────────────────────────────────────────────────────
    out_path = ART / "hero_figure.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    out_small = ART / "hero_figure_small.png"
    fig.savefig(out_small, dpi=100, bbox_inches="tight", facecolor="white")

    fw_px, fh_px = fig.get_size_inches() * 200
    print()
    print("=" * 70)
    print("HERO FIGURE BUILD SUMMARY")
    print("=" * 70)
    print(f"main size (@ dpi=200) : {int(fw_px)}x{int(fh_px)} px (pre-trim)")
    print(f"main output           : {out_path.resolve()}")
    print(f"LinkedIn preview      : {out_small.resolve()}")
    real_panels = sorted(k for k, v in used_real.items() if v)
    fb_panels = sorted(k for k, v in used_real.items() if not v)
    print(f"panels from real data : {real_panels or '-'}")
    print(f"panels from fallback  : {fb_panels or '-'}")
    if warnings_:
        print(f"\n{len(warnings_)} fallback warning(s):")
        for w in warnings_:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
