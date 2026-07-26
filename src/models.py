"""Forecasting models and an honest rolling-origin backtest.

Baseline first
--------------
`SeasonalNaive` -- last year's same month, plus a recent level adjustment -- is the
benchmark every other model has to beat. It is not a strawman: on strongly seasonal
monthly demand it is genuinely hard to beat, and a large share of published forecasting
improvements disappear when measured against it properly. MASE is reported against it
so "better" has a defined meaning.

Backtest
--------
Rolling origin, refitting at each step, expanding window. No shuffling, no random
k-fold: training on the future and testing on the past produces numbers that cannot
be reproduced in deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------- baselines


class SeasonalNaive:
    """y_hat(t) = y(t-12) + mean level shift over the last `adjust` months."""

    def __init__(self, period: int = 12, adjust: int = 3):
        self.period = period
        self.adjust = adjust
        self.history_: pd.Series | None = None

    def fit(self, y: pd.Series):
        self.history_ = y.copy()
        return self

    def predict_next(self) -> float:
        y = self.history_
        if len(y) <= self.period:
            return float(y.iloc[-1])
        seasonal = float(y.iloc[-self.period])
        if self.adjust and len(y) > self.period + self.adjust:
            recent = y.iloc[-self.adjust:].mean()
            year_ago = y.iloc[-self.period - self.adjust: -self.period].mean()
            seasonal += float(recent - year_ago)
        return seasonal


class DriftNaive:
    """Last value plus average per-period drift. A second sanity floor."""

    def __init__(self):
        self.history_: pd.Series | None = None

    def fit(self, y: pd.Series):
        self.history_ = y.copy()
        return self

    def predict_next(self) -> float:
        y = self.history_
        if len(y) < 2:
            return float(y.iloc[-1])
        drift = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        return float(y.iloc[-1] + drift)


# ------------------------------------------------------------------ ML estimators


def ridge_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 40))),
    ])


def gbm_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=300, learning_rate=0.05,
        min_samples_leaf=8, l2_regularization=1.0, random_state=0,
    )


# --------------------------------------------------------------------- backtest


@dataclass
class BacktestResult:
    predictions: pd.DataFrame        # index=date, columns=model names + actual
    metrics: pd.DataFrame

    def __repr__(self):
        return f"BacktestResult({len(self.predictions)} origins)\n{self.metrics.round(3)}"


def rolling_origin_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    min_train: int = 72,
    horizon: int = 1,
    models: dict | None = None,
) -> BacktestResult:
    """Expanding-window backtest, refitting at every origin.

    At each origin the model sees only data strictly before the forecast month, is
    refitted from scratch, and produces one `horizon`-step-ahead forecast.
    """
    models = models or {"ridge": ridge_model, "gbm": gbm_model}

    dates, actuals = [], []
    preds: dict[str, list] = {name: [] for name in models}
    preds["seasonal_naive"] = []
    preds["drift_naive"] = []

    for i in range(min_train, len(X) - horizon + 1):
        train_X, train_y = X.iloc[:i], y.iloc[:i]
        test_X = X.iloc[i + horizon - 1: i + horizon]
        if len(test_X) == 0:
            break

        dates.append(test_X.index[0])
        actuals.append(float(y.iloc[i + horizon - 1]))

        for name, ctor in models.items():
            m = ctor().fit(train_X.to_numpy(float), train_y.to_numpy(float))
            preds[name].append(float(m.predict(test_X.to_numpy(float))[0]))

        preds["seasonal_naive"].append(SeasonalNaive().fit(train_y).predict_next())
        preds["drift_naive"].append(DriftNaive().fit(train_y).predict_next())

    out = pd.DataFrame({"actual": actuals, **preds}, index=pd.DatetimeIndex(dates, name="date"))
    return BacktestResult(out, evaluate(out))


# ---------------------------------------------------------------------- metrics


def evaluate(pred_df: pd.DataFrame, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """RMSE / MAE / MAPE / MASE plus direction accuracy.

    MASE is scaled by the seasonal-naive error on the same evaluation window, so
    MASE < 1 means "better than repeating last year" and nothing else.
    """
    actual = pred_df["actual"].to_numpy()
    base_err = np.mean(np.abs(actual - pred_df[baseline].to_numpy()))

    rows = []
    for col in pred_df.columns:
        if col == "actual":
            continue
        p = pred_df[col].to_numpy()
        err = p - actual
        # Direction: did we call the month-over-month move correctly?
        d_actual = np.sign(np.diff(actual))
        d_pred = np.sign(p[1:] - actual[:-1])
        rows.append({
            "model": col,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "mape_pct": float(np.mean(np.abs(err / actual)) * 100),
            "mase": float(np.mean(np.abs(err)) / base_err) if base_err > 0 else np.nan,
            "direction_acc": float(np.mean(d_actual == d_pred)),
        })
    return pd.DataFrame(rows).sort_values("mase").reset_index(drop=True)


def threshold_events(pred_df: pd.DataFrame, model: str, window: int = 12,
                     q: float = 0.8) -> dict:
    """Can the model flag an unusually strong month before it happens?

    Point accuracy is not what a planner needs. The decision is 'should we build
    inventory' -- a classification question. Threshold is the rolling `q` quantile of
    recent actuals, computed causally.
    """
    actual = pred_df["actual"]
    thresh = actual.shift(1).rolling(window).quantile(q)
    valid = thresh.notna()

    a = (actual[valid] > thresh[valid]).to_numpy()
    p = (pred_df[model][valid] > thresh[valid]).to_numpy()

    tp = int((a & p).sum()); fp = int((~a & p).sum())
    fn = int((a & ~p).sum()); tn = int((~a & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "model": model, "n": int(valid.sum()),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def residual_intervals(pred_df: pd.DataFrame, model: str, level: float = 0.8,
                       min_hist: int = 12) -> pd.DataFrame:
    """Prediction intervals from the model's own expanding backtest residuals.

    No distributional assumption: the interval at each origin is the empirical
    quantile of errors the model has already made, using only earlier origins. It is
    the simplest honest interval available in a rolling backtest.
    """
    resid = (pred_df[model] - pred_df["actual"]).to_numpy()
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    los, his = [], []
    for i in range(len(resid)):
        if i < min_hist:
            los.append(np.nan); his.append(np.nan); continue
        past = resid[:i]
        los.append(pred_df[model].iloc[i] - np.quantile(past, hi_q))
        his.append(pred_df[model].iloc[i] - np.quantile(past, lo_q))
    out = pd.DataFrame({"lo": los, "hi": his}, index=pred_df.index)
    out["actual"] = pred_df["actual"]
    covered = (out["actual"] >= out["lo"]) & (out["actual"] <= out["hi"])
    # Rows before the warm-up have no interval. Leave them NaN rather than False --
    # a NaN comparison silently yields False, which would count every warm-up row as
    # a miss and understate coverage.
    out["covered"] = covered.where(out["lo"].notna())
    return out
