"""
Synthetic monthly demand series for a dealer-distributed durable good.

The scenario
------------
A manufacturer sells equipment through a dealer network. Monthly unit sales are
driven by seasonality, a slow trend, and the economic conditions of the customers
who buy the equipment. We want to forecast next month's units.

What makes this hard, and what the generator encodes
----------------------------------------------------

1. **External drivers act with a lead time.** A commodity price move does not change
   equipment orders this month -- it changes them two to six months later, once the
   customer's cash position and planting/herd decisions respond. The *true* lags are
   set here and hidden from the modelling code, so lag selection is a real problem.

2. **The same input has opposite signs for different customer segments.** Grain price
   is *revenue* for a crop grower and a *feed cost* for a livestock operation. Feeding
   the raw price to a linear model forces it to choose one sign, which produces the
   classic multicollinearity sign-flip. The fix is to build segment-specific *margin*
   features that are signed correctly by construction.

3. **Internal operating aggregates are contemporaneous, not predictive.** Counts like
   active dealers and units on lot are only known for the same month as the sales they
   describe. They correlate strongly with demand -- because they are partly caused by
   it -- so using them unlagged leaks the answer into the features.

4. **Published economic series arrive late.** Official statistics for month M are
   released during month M+1. A backtest that reads them as if available in month M
   is quietly using data that did not exist yet.

All randomness is seeded; `make_dataset()` is fully reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_MONTHS = 168          # 14 years of monthly observations
START = "2012-01-01"

# Ground-truth lead times, in months. The modelling code never sees these.
# Keyed by the *constructed causal driver*, not the raw price, because that is the
# quantity that actually moves a purchase decision (see features.add_segment_margins).
TRUE_LAGS = {
    "crop_margin": 4,
    "livestock_margin": 5,
    "interest_rate": 6,
    "sentiment": 2,
    "precipitation": 3,
}

# Publication delay, in months, for officially released series.
PUBLICATION_DELAY = {
    "grain_price": 1,
    "livestock_price": 1,
    "feed_cost": 1,
    "interest_rate": 0,     # market rate, available same day
    "sentiment": 0,
    "precipitation": 0,
}


def _ar1(rng, n, phi, sd, mean=0.0):
    """Persistent economic series -- prices do not wander like white noise."""
    x = np.zeros(n)
    x[0] = rng.normal(mean, sd / np.sqrt(1 - phi**2))
    for t in range(1, n):
        x[t] = mean * (1 - phi) + phi * x[t - 1] + rng.normal(0, sd)
    return x


def make_dataset(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=N_MONTHS, freq="MS")
    t = np.arange(N_MONTHS)

    # ---------------------------------------------------------- external drivers
    grain_price = 380 + 90 * _ar1(rng, N_MONTHS, 0.93, 0.30)          # cents/bushel
    livestock_price = 128 + 18 * _ar1(rng, N_MONTHS, 0.94, 0.28)      # cents/lb
    feed_cost = 190 + 34 * _ar1(rng, N_MONTHS, 0.90, 0.32)            # $/ton
    # Feed cost genuinely tracks grain, which is what creates the collinearity.
    feed_cost += 0.10 * (grain_price - grain_price.mean())

    interest_rate = np.clip(3.2 + 2.4 * _ar1(rng, N_MONTHS, 0.97, 0.22), 0.4, 11.0)
    sentiment = 92 + 11 * _ar1(rng, N_MONTHS, 0.88, 0.36)

    # Precipitation: seasonal with year-to-year variation.
    month = idx.month.to_numpy()
    precipitation = (
        72 + 34 * np.sin(2 * np.pi * (month - 4) / 12) + rng.normal(0, 15, N_MONTHS)
    ).clip(4, None)

    # ------------------------------------------------- customer segment margins
    # These are the *causal* quantities. A crop grower's margin rises with grain
    # price; a livestock operation's margin falls with it, because grain is feed.
    crop_margin = 0.055 * (grain_price - 380) - 0.020 * (feed_cost - 190)
    livestock_margin = 0.30 * (livestock_price - 128) - 0.055 * (feed_cost - 190)

    def lag(a, k):
        out = np.full_like(a, np.nan, dtype=float)
        if k > 0:
            out[k:] = a[:-k]
        else:
            out[:] = a
        return out

    # ---------------------------------------------------------------- demand
    seasonal = 165 * np.sin(2 * np.pi * (month - 3) / 12) + 55 * np.sin(4 * np.pi * (month - 1) / 12)
    trend = 0.9 * t
    cycle = 60 * np.sin(2 * np.pi * t / 62)

    driver = (
        4.2 * np.nan_to_num(lag(crop_margin, TRUE_LAGS["crop_margin"]))
        + 3.4 * np.nan_to_num(lag(livestock_margin, TRUE_LAGS["livestock_margin"]))
        - 26.0 * np.nan_to_num(lag(interest_rate - 3.2, TRUE_LAGS["interest_rate"]))
        + 3.1 * np.nan_to_num(lag(sentiment - 92, TRUE_LAGS["sentiment"]))
        + 0.55 * np.nan_to_num(lag(precipitation - 72, TRUE_LAGS["precipitation"]))
    )

    base = 980 + trend + cycle + seasonal + driver
    noise = rng.normal(0, 46, N_MONTHS)
    units_sold = np.clip(base + noise, 40, None).round().astype(int)

    # --------------------------------------- contemporaneous internal aggregates
    # Partly *caused by* the same month's demand -- the leakage trap.
    unique_dealers = np.clip(
        58 + 0.020 * units_sold + rng.normal(0, 3.0, N_MONTHS), 5, None
    ).round().astype(int)
    stock_on_lot = np.clip(
        1500 - 0.42 * units_sold + 55 * np.sin(2 * np.pi * (month - 1) / 12)
        + rng.normal(0, 60, N_MONTHS), 30, None
    ).round().astype(int)
    avg_days_on_lot = np.clip(
        120 - 0.030 * units_sold + rng.normal(0, 7, N_MONTHS), 5, None
    ).round(1)

    df = pd.DataFrame(
        {
            "date": idx,
            "units_sold": units_sold,
            # external, as published (delay applied below)
            "grain_price": grain_price.round(1),
            "livestock_price": livestock_price.round(2),
            "feed_cost": feed_cost.round(1),
            "interest_rate": interest_rate.round(3),
            "sentiment": sentiment.round(1),
            "precipitation": precipitation.round(1),
            # internal, contemporaneous
            "unique_dealers": unique_dealers,
            "stock_on_lot": stock_on_lot,
            "avg_days_on_lot": avg_days_on_lot,
            # oracle columns for teaching -- dropped before modelling
            "_crop_margin": crop_margin.round(3),
            "_livestock_margin": livestock_margin.round(3),
            "_signal_no_noise": base.round(2),
        }
    ).set_index("date")

    return df


def apply_publication_delay(df: pd.DataFrame) -> pd.DataFrame:
    """Shift officially published series by their release delay.

    Without this the backtest reads statistics for month M during month M, which is
    a month before they actually exist. It is a small change that costs real accuracy
    -- and omitting it is one of the most common ways a forecasting backtest flatters
    itself.
    """
    out = df.copy()
    for col, d in PUBLICATION_DELAY.items():
        if d > 0 and col in out.columns:
            out[col] = out[col].shift(d)
    return out


ORACLE_COLS = ["_crop_margin", "_livestock_margin", "_signal_no_noise"]
INTERNAL_CONTEMPORANEOUS = ["unique_dealers", "stock_on_lot", "avg_days_on_lot"]
EXTERNAL_COLS = ["grain_price", "livestock_price", "feed_cost",
                 "interest_rate", "sentiment", "precipitation"]
# The columns lag selection should actually scan: constructed margins plus the
# drivers that act directly.
DRIVER_COLS = ["crop_margin", "livestock_margin", "interest_rate",
               "sentiment", "precipitation"]
TARGET = "units_sold"


def drop_oracle(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ORACLE_COLS if c in df.columns])


if __name__ == "__main__":
    import pathlib

    d = make_dataset()
    out = pathlib.Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    d.to_csv(out / "demand.csv")
    print(d.shape)
    print(d[["units_sold", "grain_price", "interest_rate", "unique_dealers"]].describe().round(1))
    snr = 1 - np.var(d["units_sold"] - d["_signal_no_noise"]) / np.var(d["units_sold"])
    print(f"oracle R2 ceiling: {snr:.3f}")
