"""
quant/ml/transformer_model.py — TFT-inspired Transformer for return forecasting.

A lightweight PyTorch implementation of a multi-head attention encoder operating
on a rolling window of features per (ticker, date). The model learns which
past timesteps and which features matter most for the forward 21-day return.

Design choices (kept small so the project is reproducible on a laptop):
    - sequence length L = 60 trading days
    - static covariate: ticker embedding (captures cross-sectional structure)
    - d_model = 64, 4 heads, 2 encoder layers
    - causal attention (no future leakage)
    - final MLP head outputs a scalar: predicted forward log-return

The full Temporal Fusion Transformer (Lim et al., 2021) is a superset of this;
we strip it down to the attention backbone + static embedding so training
stays fast without sacrificing the interesting bits (multi-head attention,
learned per-timestep weights, per-ticker embeddings).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Torch import is deferred so the module is importable without torch ──
def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:
        raise ImportError(
            "torch is required for the Transformer model. "
            "Install it via `pip install torch`."
        ) from e


@dataclass
class TransformerConfig:
    seq_len: int = 60
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    ticker_embed_dim: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 15
    device: str = "cpu"
    random_seed: int = 42

    # Ablation / architecture switches
    use_ticker_embed: bool = True       # ablation: toggle static ticker embedding
    use_causal_mask: bool = True         # ablation: toggle causal mask
    pooling: str = "last"                # "last" | "attn" (attention-weighted)

    # Multi-horizon forecasting
    target_cols: tuple = ("target_fwd_ret_21d",)   # one head per target
    primary_target: str = "target_fwd_ret_21d"     # used by conformal / eval

    # Loss function: "mse" (point regression) or "pairwise" (rank-based).
    # Rationale: we use predictions to RANK tickers and go long top /
    # short bottom — MSE on the target is a proxy for ranking quality at
    # best. Pairwise hinge directly optimises the ranking, and typically
    # improves IC by 2-3pp on financial forecasting benchmarks.
    loss_fn: str = "mse"
    pairwise_margin: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Sequence-dataset construction
# ──────────────────────────────────────────────────────────────────────────────

def _build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    seq_len: int,
    require_target: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]], Dict[str, int]]:
    """
    Build (X_seq, X_static, Y, sample_dates, sample_tickers) arrays.

    For each (ticker, date_t) sample, X_seq is the (seq_len × n_features)
    window ending at date_t, X_static is the ticker id, and Y is a
    (n_targets,) vector of forward targets at date_t.

    require_target: if True, drop rows where any target is NaN (train/calib).
    If False, fill missing targets with 0 and emit the row anyway (predict).
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    tickers = sorted(df["ticker"].unique().tolist())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}

    X_seq: List[np.ndarray] = []
    X_static: List[int] = []
    Y: List[np.ndarray] = []
    sample_dates: List[pd.Timestamp] = []
    sample_tickers: List[str] = []

    feat_arr_dtype = np.float32

    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        feats = grp[feature_cols].astype(feat_arr_dtype).values
        present_cols = [c for c in target_cols if c in grp.columns]
        if not present_cols:
            targets = np.full((len(grp), len(target_cols)), np.nan, dtype=feat_arr_dtype)
        else:
            targets = grp[present_cols].astype(feat_arr_dtype).values
            # Pad missing columns with NaN to keep shape stable
            if len(present_cols) < len(target_cols):
                pad = np.full((len(grp), len(target_cols) - len(present_cols)), np.nan, dtype=feat_arr_dtype)
                targets = np.concatenate([targets, pad], axis=1)
        dates = pd.to_datetime(grp["date"]).values

        for t in range(seq_len - 1, len(grp)):
            window = feats[t - seq_len + 1 : t + 1]
            if np.isnan(window).any():
                continue
            y_vec = targets[t].copy()
            if require_target and np.isnan(y_vec).any():
                continue
            if not require_target:
                y_vec = np.where(np.isnan(y_vec), 0.0, y_vec)
            X_seq.append(window)
            X_static.append(ticker_to_id[ticker])
            Y.append(y_vec)
            sample_dates.append(pd.Timestamp(dates[t]))
            sample_tickers.append(ticker)

    if not X_seq:
        arrays = (
            np.zeros((0, seq_len, len(feature_cols)), dtype=feat_arr_dtype),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, len(target_cols)), dtype=feat_arr_dtype),
            np.array([], dtype="datetime64[ns]"),
            [],
        )
        return arrays, ticker_to_id

    arrays = (
        np.stack(X_seq).astype(feat_arr_dtype),
        np.asarray(X_static, dtype=np.int64),
        np.stack(Y).astype(feat_arr_dtype),
        np.asarray(sample_dates, dtype="datetime64[ns]"),
        sample_tickers,
    )
    return arrays, ticker_to_id


# ──────────────────────────────────────────────────────────────────────────────
# Model definition (lazy — only imported once torch is available)
# ──────────────────────────────────────────────────────────────────────────────

def _define_model(cfg: TransformerConfig, n_features: int, n_tickers: int, n_targets: int):
    torch = _torch()
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class ReturnTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.feature_proj = nn.Linear(n_features, cfg.d_model)
            self.pos_enc = PositionalEncoding(cfg.d_model, cfg.seq_len + 4)
            if cfg.use_ticker_embed:
                self.ticker_embed = nn.Embedding(n_tickers, cfg.ticker_embed_dim)
                static_dim = cfg.ticker_embed_dim
            else:
                self.ticker_embed = None
                static_dim = 0
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
            # Attention-weighted pooling (one scalar score per timestep)
            if cfg.pooling == "attn":
                self.attn_pool = nn.Linear(cfg.d_model, 1)
            else:
                self.attn_pool = None
            # Multi-horizon head: one output per target
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model + static_dim, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, n_targets),
            )
            # Most recent attention weights captured during forward (B, L)
            self.last_attn_weights: "torch.Tensor | None" = None

        @staticmethod
        def _causal_mask(L: int, device) -> "torch.Tensor":  # noqa: F821
            mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
            return mask

        def forward(self, x_seq, x_static):
            # x_seq: (B, L, F); x_static: (B,)
            h = self.feature_proj(x_seq)
            h = self.pos_enc(h)
            L = h.size(1)
            mask = self._causal_mask(L, h.device) if self.cfg.use_causal_mask else None
            h = self.encoder(h, mask=mask)

            if self.attn_pool is not None:
                # Attention-weighted pooling: softmax over timesteps
                scores = self.attn_pool(h).squeeze(-1)     # (B, L)
                weights = torch.softmax(scores, dim=-1)    # (B, L)
                pooled = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
            else:
                # Last-token pooling (also capture a "pseudo-attention" of a
                # one-hot on the last timestep so downstream viz code always has
                # something to plot)
                pooled = h[:, -1, :]
                weights = torch.zeros(h.size(0), h.size(1), device=h.device)
                weights[:, -1] = 1.0

            self.last_attn_weights = weights.detach()

            if self.ticker_embed is not None:
                emb = self.ticker_embed(x_static)
                feat = torch.cat([pooled, emb], dim=-1)
            else:
                feat = pooled
            out = self.head(feat)        # (B, n_targets)
            return out

    return ReturnTransformer()


# ──────────────────────────────────────────────────────────────────────────────
# Public wrapper — fit / predict
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerForecaster:
    feature_cols: List[str]
    target_col: str = "target_fwd_ret_21d"    # primary target used by conformal/eval
    config: TransformerConfig = field(default_factory=TransformerConfig)

    # Populated on fit()
    model: Optional[object] = None
    ticker_to_id: Dict[str, int] = field(default_factory=dict)
    feat_mean: Optional[np.ndarray] = None
    feat_std: Optional[np.ndarray] = None

    @property
    def _targets(self) -> List[str]:
        return list(self.config.target_cols) or [self.target_col]

    @property
    def _primary_idx(self) -> int:
        return self._targets.index(self.config.primary_target)

    # ── Fit ──
    def fit(self, train_df: pd.DataFrame) -> "TransformerForecaster":
        torch = _torch()
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Standardise features with train-time stats (persisted for predict)
        train_df = train_df.copy()
        feats = train_df[self.feature_cols].astype(np.float32).values
        self.feat_mean = np.nanmean(feats, axis=0)
        self.feat_std = np.nanstd(feats, axis=0)
        self.feat_std[self.feat_std < 1e-8] = 1.0
        train_df[self.feature_cols] = (feats - self.feat_mean) / self.feat_std

        (Xs, Xst, Y, dates_arr, _tks), tkmap = _build_sequences(
            train_df, self.feature_cols, self._targets, self.config.seq_len,
            require_target=True,
        )
        self.ticker_to_id = tkmap

        if len(Xs) == 0:
            raise ValueError("No training sequences produced — check history length.")

        device = torch.device(self.config.device)
        model = _define_model(
            self.config, len(self.feature_cols), len(tkmap), len(self._targets)
        ).to(device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
        Xst_t = torch.tensor(Xst, dtype=torch.long, device=device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
        # Integer-encode dates so we can group a batch into per-date sub-lists
        # for the pairwise loss without going back to pandas.
        uniq_dates, date_ids = np.unique(dates_arr, return_inverse=True)
        date_ids_t = torch.tensor(date_ids, dtype=torch.long, device=device)

        mse = torch.nn.MSELoss()
        primary_idx = self._primary_idx
        margin = float(self.config.pairwise_margin)

        def pairwise_loss(pred_primary: "torch.Tensor",
                          y_primary: "torch.Tensor",
                          d_ids: "torch.Tensor") -> "torch.Tensor":
            """
            Hinge-style pairwise ranking loss on the primary target.
            For each date, form pairs (i,j) and penalise when the ordering
            of predictions disagrees with the ordering of realised returns.
            """
            total = pred_primary.new_zeros(())
            total_pairs = 0
            # Iterate unique dates present in this batch
            for d in torch.unique(d_ids):
                mask = d_ids == d
                if mask.sum() < 2:
                    continue
                y_d = y_primary[mask]
                p_d = pred_primary[mask]
                # Pairs where y_i > y_j → want p_i > p_j + margin
                diff_y = y_d.unsqueeze(1) - y_d.unsqueeze(0)       # (n,n)
                diff_p = p_d.unsqueeze(1) - p_d.unsqueeze(0)
                # Only upper triangle (i<j) with y_i != y_j
                n_d = y_d.shape[0]
                triu = torch.triu(torch.ones(n_d, n_d, device=y_d.device), diagonal=1).bool()
                mask_pairs = triu & (diff_y.abs() > 1e-8)
                if mask_pairs.sum() == 0:
                    continue
                sign = torch.sign(diff_y)
                # Hinge loss: max(0, margin - sign(diff_y) * diff_p)
                hinge = torch.clamp(margin - sign * diff_p, min=0.0)
                total = total + hinge[mask_pairs].sum()
                total_pairs += int(mask_pairs.sum().item())
            if total_pairs == 0:
                return pred_primary.new_zeros(())
            return total / total_pairs

        n = len(Xs_t)
        bs = self.config.batch_size
        model.train()
        for epoch in range(self.config.epochs):
            perm = torch.randperm(n, device=device)
            total = 0.0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                pred = model(Xs_t[idx], Xst_t[idx])     # (b, n_targets)
                y_batch = Y_t[idx]
                if self.config.loss_fn == "pairwise":
                    # Combine a light MSE term for stabilisation + pairwise on primary
                    reg_loss = mse(pred, y_batch) * 0.1
                    rank_loss = pairwise_loss(
                        pred[:, primary_idx], y_batch[:, primary_idx], date_ids_t[idx]
                    )
                    loss = reg_loss + rank_loss
                else:
                    loss = mse(pred, y_batch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.item()) * len(idx)
            if (epoch + 1) % max(1, self.config.epochs // 5) == 0:
                logger.info("  epoch %2d/%d  loss(%s)=%.6f",
                            epoch + 1, self.config.epochs,
                            self.config.loss_fn, total / n)

        model.eval()
        self.model = model
        return self

    # ── Predict ──
    def predict(
        self,
        df: pd.DataFrame,
        context_df: Optional[pd.DataFrame] = None,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          ticker, date, prediction,
          pred_<target> for each target in config.target_cols.
        If return_attention=True, also include 'attn' (list[float] per row,
        softmax over the seq_len timesteps).

        Parameters
        ----------
        df : pd.DataFrame
            Rows for which predictions are desired.
        context_df : pd.DataFrame, optional
            Earlier rows (same columns) used only to build the seq_len window
            that ends at each prediction date. If None, `df` must include
            enough historical rows per ticker.
        """
        if self.model is None:
            raise RuntimeError("Call .fit() first")

        torch = _torch()
        device = torch.device(self.config.device)

        target_dates = set(pd.to_datetime(df["date"]).unique())

        full = df if context_df is None else pd.concat([context_df, df], ignore_index=True)
        full = full.drop_duplicates(subset=["ticker", "date"], keep="last")
        full = full[full["ticker"].isin(self.ticker_to_id)]

        feats = full[self.feature_cols].astype(np.float32).values
        full = full.copy()
        full[self.feature_cols] = (feats - self.feat_mean) / self.feat_std

        out = _build_sequences(
            full, self.feature_cols, self._targets, self.config.seq_len,
            require_target=False,
        )
        (Xs, Xst, _Y, dates, tks), _ = out
        if len(Xs) == 0:
            cols = ["ticker", "date", "prediction"] + [f"pred_{t}" for t in self._targets]
            if return_attention:
                cols.append("attn")
            return pd.DataFrame(columns=cols)

        Xst = np.array([self.ticker_to_id[t] for t in tks], dtype=np.int64)

        with torch.no_grad():
            Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
            Xst_t = torch.tensor(Xst, dtype=torch.long, device=device)
            preds = self.model(Xs_t, Xst_t).cpu().numpy()
            attn = self.model.last_attn_weights.cpu().numpy() if return_attention else None

        res = pd.DataFrame({"ticker": tks, "date": pd.to_datetime(dates)})
        for i, tgt in enumerate(self._targets):
            res[f"pred_{tgt}"] = preds[:, i]
        # Primary prediction (used by conformal wrapper, eval)
        res["prediction"] = res[f"pred_{self.config.primary_target}"]

        if return_attention:
            res["attn"] = [a.tolist() for a in attn]

        mask = res["date"].isin(target_dates)
        return res[mask].reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    from quant.ml.features import build_feature_panel, FEATURE_COLS
    prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
    panel = build_feature_panel(prices)
    cutoff = panel["date"].quantile(0.8)
    train = panel[panel["date"] < cutoff]
    test = panel[panel["date"] >= cutoff]
    cfg = TransformerConfig(
        epochs=5,
        target_cols=("target_fwd_ret_5d", "target_fwd_ret_21d", "target_fwd_ret_63d"),
        primary_target="target_fwd_ret_21d",
        pooling="attn",
    )
    fc = TransformerForecaster(feature_cols=FEATURE_COLS, config=cfg).fit(train)
    preds = fc.predict(test, return_attention=True)
    print(preds.drop(columns=["attn"], errors="ignore").tail())
