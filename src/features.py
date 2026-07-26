"""Feature construction, including the segment-margin fix for sign flips.

The sign-flip problem
---------------------
Grain price enters the business through two opposing channels. For a crop grower it
is revenue; for a livestock operation it is a feed cost. Both segments buy the same
equipment. Handed the raw price, a linear model must pick one coefficient sign for a
variable whose true effect is positive for some customers and negative for others --
and because grain and feed cost are strongly collinear, the fitted signs become
unstable and flip between specifications.

The fix is not a regularisation trick. It is to build the quantity that actually
drives the purchase decision -- **segment margin** -- so each feature is signed
correctly by construction:

    crop_margin      ~  grain revenue  -  input cost
    livestock_margin ~  livestock revenue  -  feed cost

This mirrors how published industry margin measures are built (e.g. USDA's Dairy
Margin Coverage formula, or the CME cattle crush spread): revenue leg minus cost leg,
in consistent units. The model then sees one coefficient per segment with a stable,
interpretable sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generate import EXTERNAL_COLS, INTERNAL_CONTEMPORANEOUS, TARGET


def add_segment_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Composite margin features, signed correctly by construction.

    Coefficients are the published-formula weights, not fitted parameters -- the
    point is that these are *domain* constructions, decided before any model runs.
    """
    out = df.copy()
    out["crop_margin"] = 0.055 * (out["grain_price"] - 380) - 0.020 * (out["feed_cost"] - 190)
    out["livestock_margin"] = 0.30 * (out["livestock_price"] - 128) - 0.055 * (out["feed_cost"] - 190)
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    out["month_sin2"] = np.sin(4 * np.pi * m / 12)
    out["month_cos2"] = np.cos(4 * np.pi * m / 12)
    out["time_index"] = np.arange(len(out))
    return out


def add_target_history(df: pd.DataFrame, lags=(1, 2, 3, 12), rolls=(3, 6, 12)) -> pd.DataFrame:
    """Lags and rolling statistics of the target.

    Every one is shifted by at least one period. A rolling mean computed on a window
    that includes the current month is a direct leak, and it is an easy one to write
    by accident: `.rolling(3).mean()` without a `.shift(1)` silently includes today.
    """
    out = df.copy()
    y = out[TARGET]
    for L in lags:
        out[f"{TARGET}_lag{L}"] = y.shift(L)
    for w in rolls:
        out[f"{TARGET}_roll{w}_mean"] = y.shift(1).rolling(w).mean()
        out[f"{TARGET}_roll{w}_std"] = y.shift(1).rolling(w).std()
    # Year-over-year change of the lagged level -- momentum without touching today.
    out[f"{TARGET}_yoy"] = y.shift(1) / y.shift(13) - 1.0
    return out


def add_external_lags(df: pd.DataFrame, lag_map: dict[str, int],
                      momentum: bool = True) -> pd.DataFrame:
    """Apply a chosen lag per external driver, plus optional momentum terms."""
    out = df.copy()
    for col, k in lag_map.items():
        if col not in out.columns:
            continue
        out[f"{col}_lag{k}"] = out[col].shift(k)
        if momentum:
            out[f"{col}_mom3_lag{k}"] = out[col].shift(k) - out[col].shift(k + 3)
    return out


def build_matrix(
    df: pd.DataFrame,
    lag_map: dict[str, int],
    *,
    include_contemporaneous_internal: bool,
    internal_lag: int = 1,
    drop_raw_external: bool = True,
) -> pd.DataFrame:
    """Assemble the model matrix.

    Parameters
    ----------
    include_contemporaneous_internal
        If True, operating aggregates enter unlagged -- the leak. If False they are
        shifted by `internal_lag`, which is the only honest option: they describe the
        month they are measured in, and that month has not happened yet at forecast time.
    drop_raw_external
        Whether to remove the unlagged external columns after lagged versions are built.
        Leaving them in is another silent leak, since the raw column for month M is not
        knowable when forecasting month M.
    """
    out = add_segment_margins(df)
    out = add_calendar(out)
    out = add_target_history(out)
    out = add_external_lags(out, lag_map)

    for c in INTERNAL_CONTEMPORANEOUS:
        if c not in out.columns:
            continue
        if include_contemporaneous_internal:
            pass                                   # left as-is: the leak
        else:
            out[c] = out[c].shift(internal_lag)

    if drop_raw_external:
        drop = [c for c in EXTERNAL_COLS if c in out.columns]
        drop += [c for c in ("crop_margin", "livestock_margin") if c in out.columns]
        out = out.drop(columns=drop)

    return out


def xy(mat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features and target, dropping rows with any missing feature."""
    y = mat[TARGET]
    X = mat.drop(columns=[TARGET])
    keep = X.notna().all(axis=1) & y.notna()
    return X.loc[keep], y.loc[keep]
