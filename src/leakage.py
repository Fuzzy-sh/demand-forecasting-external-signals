"""Lag selection and the leakage audit.

Two distinct leaks live in this file, and they are easy to confuse.

**Leak 1 -- feature timing.** Using a value that will not be known at forecast time:
contemporaneous internal aggregates, unlagged external drivers, a rolling window that
includes the current period. Handled in `features.py` and audited by
`audit_feature_timing()`.

**Leak 2 -- selection.** Choosing *which* lag to use by scanning correlations over the
whole series, including the test period. No individual feature is then mistimed, but
the choice of features was informed by data the model is about to be scored on. This
is subtler, survives most code review, and inflates backtests on its own.

`select_lags()` takes an explicit `train_end` for exactly this reason: the scan can
only see the training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generate import EXTERNAL_COLS, INTERNAL_CONTEMPORANEOUS, TARGET


def select_lags(
    df: pd.DataFrame,
    columns: list[str],
    *,
    train_end: pd.Timestamp | None,
    max_lag: int = 12,
    horizon: int = 1,
    deseasonalize: bool = True,
    seasonal_period: int = 12,
) -> pd.DataFrame:
    """Pick the lag maximising |corr(feature.shift(k), target)| for each column.

    Parameters
    ----------
    train_end
        Last date the scan is allowed to see. Pass ``None`` to scan the full series --
        which is the selection leak, reproduced deliberately so its cost can be measured.
    horizon
        Forecast horizon in periods. The target is shifted forward so the scan asks
        "does this feature lead the value I will have to predict?"
    deseasonalize
        Seasonally difference both sides before correlating. Without this the target's
        annual cycle dominates every correlation and the scan recovers the seasonal
        period rather than the driver's lead time -- it will happily report a lag of 12
        for a variable whose true lead is 4. Seasonal differencing also removes the
        trend, leaving the cyclical component the external drivers actually explain.
    """
    scan = df if train_end is None else df.loc[:train_end]
    y = scan[TARGET].shift(-horizon)
    if deseasonalize:
        y = y - y.shift(seasonal_period)

    rows = []
    for col in columns:
        if col not in scan.columns:
            continue
        best_k, best_c = 0, 0.0
        for k in range(0, max_lag + 1):
            f = scan[col].shift(k)
            if deseasonalize:
                f = f - f.shift(seasonal_period)
            c = f.corr(y)
            if np.isfinite(c) and abs(c) > abs(best_c):
                best_k, best_c = k, c
        rows.append({"feature": col, "selected_lag": best_k, "abs_corr": abs(best_c),
                     "corr": best_c})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def lag_map(sel: pd.DataFrame, minimum: int = 1) -> dict[str, int]:
    """Selection table -> {feature: lag}, with a floor.

    The floor matters. A scan will often return lag 0 for a driver that moves with
    demand, but lag 0 means "this month's value", which is not available when the
    forecast is made. Anything the scan wants at lag 0 gets pushed to `minimum`.
    """
    return {r.feature: max(int(r.selected_lag), minimum) for r in sel.itertuples()}


def audit_feature_timing(X: pd.DataFrame, y: pd.Series, top_n: int = 12) -> pd.DataFrame:
    """Flag features suspiciously correlated with the *current* target.

    A genuine leading indicator correlates with the future. A leaked feature correlates
    with the present more strongly than with anything it should lead. The audit compares
    each feature's correlation with y(t) against its correlation with y(t+1): when the
    contemporaneous relationship dominates, the feature is describing the present rather
    than predicting the future.
    """
    rows = []
    y_next = y.shift(-1)
    for c in X.columns:
        r_now = X[c].corr(y)
        r_next = X[c].corr(y_next)
        rows.append({
            "feature": c,
            "corr_with_y_t": r_now,
            "corr_with_y_t+1": r_next,
            "contemporaneous_excess": abs(r_now) - abs(r_next),
        })
    out = pd.DataFrame(rows).sort_values("contemporaneous_excess", ascending=False)
    out["suspect"] = (out["contemporaneous_excess"] > 0.05) & (out["corr_with_y_t"].abs() > 0.3)
    return out.reset_index(drop=True).head(top_n) if top_n else out.reset_index(drop=True)


def describe_leaks() -> pd.DataFrame:
    """The catalogue of leaks this project guards against, and the guard used."""
    return pd.DataFrame([
        {"leak": "contemporaneous internal aggregates",
         "example": "unique_dealers, stock_on_lot for the month being forecast",
         "guard": "shift by >= 1 period before use"},
        {"leak": "unlagged external drivers",
         "example": "grain_price for month M used to forecast month M",
         "guard": "drop raw columns; keep only explicitly lagged versions"},
        {"leak": "rolling window including the current period",
         "example": "y.rolling(3).mean() without .shift(1)",
         "guard": "shift(1) before every rolling call"},
        {"leak": "publication delay ignored",
         "example": "official statistics read a month before release",
         "guard": "apply_publication_delay() before feature building"},
        {"leak": "lag chosen on the full series",
         "example": "scanning corr over train+test to pick each feature's lag",
         "guard": "select_lags(train_end=...) restricts the scan to training data"},
        {"leak": "random shuffled cross-validation",
         "example": "KFold on a time series, training on the future",
         "guard": "rolling-origin backtest only"},
    ])
