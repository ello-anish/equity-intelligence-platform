"""
quant.ml — Machine-learning layer for the Equity Intelligence Platform.

Modules
-------
features         : engineered cross-sectional & time-series features
regimes          : Hidden-Markov-Model regime detection on market state
walkforward      : purged walk-forward CV splitter (Lopez de Prado style)
baseline         : scikit-learn GradientBoosting benchmark model
transformer_model: PyTorch attention-based sequence forecaster (TFT-inspired)
conformal        : split-conformal prediction intervals (MAPIE-style, local impl)
shap_explain     : SHAP-based global & per-sample feature attribution
evaluation       : regime-conditional performance + conformal calibration
"""
