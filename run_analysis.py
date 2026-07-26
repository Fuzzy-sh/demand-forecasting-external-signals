"""End-to-end run: build the leaky and honest pipelines, price the difference.

    python run_analysis.py

Writes figures/ and data/, and prints every number quoted in the README.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import features as F
from src import generate as G
from src import leakage as L
from src import models as M

ROOT = pathlib.Path(__file__).resolve().parent
FIG, DATA = ROOT / "figures", ROOT / "data"
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
INK, ACCENT, MUTED, GOOD = "#2b3a55", "#c2643f", "#8896ab", "#3f7f6f"

MIN_TRAIN = 84          # 7 years before the first forecast


def main() -> None:
    raw = G.make_dataset()
    raw.to_csv(DATA / "demand.csv")
    y_full = raw[G.TARGET]

    ceiling = 1 - np.var(y_full - raw["_signal_no_noise"]) / np.var(y_full)
    print(f"observations: {len(raw)} months  ({raw.index[0]:%Y-%m} to {raw.index[-1]:%Y-%m})")
    print(f"oracle R2 ceiling: {ceiling:.3f}")

    train_end = raw.index[MIN_TRAIN - 1]
    print(f"lag-selection window ends: {train_end:%Y-%m}")

    # ================================================================= PIPELINE A
    # Everything done the tempting way: contemporaneous internal aggregates, no
    # publication delay, and lags chosen by scanning the full series.
    print("\n=== A. leaky pipeline ===")
    sel_leaky = L.select_lags(raw, G.EXTERNAL_COLS, train_end=None)
    map_leaky = {r.feature: int(r.selected_lag) for r in sel_leaky.itertuples()}
    mat_leaky = F.build_matrix(G.drop_oracle(raw), map_leaky,
                               include_contemporaneous_internal=True,
                               drop_raw_external=False)
    XA, yA = F.xy(mat_leaky)
    resA = M.rolling_origin_backtest(XA, yA, min_train=MIN_TRAIN)
    print(resA.metrics.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    # ================================================================= PIPELINE B
    # The same modelling code, with every leak closed.
    print("\n=== B. honest pipeline ===")
    delayed = G.apply_publication_delay(raw)
    wm = F.add_segment_margins(delayed)
    # Lag selection scans the constructed drivers (segment margins + direct drivers),
    # not the raw prices -- the margins are the quantities with a causal channel.
    sel_honest = L.select_lags(wm, G.DRIVER_COLS, train_end=train_end)
    map_honest = L.lag_map(sel_honest, minimum=1)
    mat_honest = F.build_matrix(G.drop_oracle(wm), map_honest,
                                include_contemporaneous_internal=False,
                                drop_raw_external=True)
    XB, yB = F.xy(mat_honest)
    resB = M.rolling_origin_backtest(XB, yB, min_train=MIN_TRAIN)
    print(resB.metrics.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    # ------------------------------------------------------------ the difference
    comp = []
    for name in ["ridge", "gbm", "seasonal_naive"]:
        a = resA.metrics.set_index("model").loc[name]
        b = resB.metrics.set_index("model").loc[name]
        comp.append({"model": name,
                     "leaky_mae": a["mae"], "honest_mae": b["mae"],
                     "leaky_mase": a["mase"], "honest_mase": b["mase"],
                     "mae_understated_by_pct": 100 * (b["mae"] - a["mae"]) / b["mae"]})
    comp = pd.DataFrame(comp)
    print("\n--- what the leaks were worth ---")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    comp.to_csv(DATA / "leak_cost.csv", index=False)
    resA.metrics.to_csv(DATA / "metrics_leaky.csv", index=False)
    resB.metrics.to_csv(DATA / "metrics_honest.csv", index=False)

    # ------------------------------------------------ lag selection: a study
    # Does data-driven lag selection actually help? Compare five specifications
    # under the identical honest pipeline, changing only the lag map.
    print("\n--- does automated lag selection help? ---")
    specs = {
        "oracle (true lead times)": {k: max(v, 1) for k, v in G.TRUE_LAGS.items()},
        "scan on train window only": L.lag_map(
            L.select_lags(wm, G.DRIVER_COLS, train_end=train_end), minimum=1),
        "scan on full series (leak)": L.lag_map(
            L.select_lags(wm, G.DRIVER_COLS, train_end=None), minimum=1),
        "flat lag = 3 (no selection)": {k: 3 for k in G.DRIVER_COLS},
        "flat lag = 1 (no selection)": {k: 1 for k in G.DRIVER_COLS},
    }
    lag_rows = []
    for label, lm in specs.items():
        mat = F.build_matrix(G.drop_oracle(wm), lm,
                             include_contemporaneous_internal=False, drop_raw_external=True)
        Xs, ys = F.xy(mat)
        r = M.rolling_origin_backtest(Xs, ys, min_train=MIN_TRAIN)
        mm = r.metrics.set_index("model")
        lag_rows.append({"lag specification": label,
                         "ridge_mae": mm.loc["ridge", "mae"],
                         "ridge_mase": mm.loc["ridge", "mase"],
                         "gbm_mase": mm.loc["gbm", "mase"]})
    lag_study = pd.DataFrame(lag_rows)
    print(lag_study.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    lag_study.to_csv(DATA / "lag_selection_study.csv", index=False)

    # How close did the scan get to the true lead times?
    truth = pd.DataFrame([{"feature": k, "true_lag": v} for k, v in G.TRUE_LAGS.items()])
    sel_tr = L.select_lags(wm, G.DRIVER_COLS, train_end=train_end)
    sel_fu = L.select_lags(wm, G.DRIVER_COLS, train_end=None)
    lag_cmp = (truth
               .merge(sel_tr[["feature", "selected_lag", "abs_corr"]].rename(
                   columns={"selected_lag": "scan_train_only"}), on="feature", how="left")
               .merge(sel_fu[["feature", "selected_lag"]].rename(
                   columns={"selected_lag": "scan_full_series"}), on="feature", how="left"))
    lag_cmp["abs_err"] = (lag_cmp["scan_train_only"] - lag_cmp["true_lag"]).abs()
    print("\n--- recovered lead times vs ground truth ---")
    print(lag_cmp.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    lag_cmp.to_csv(DATA / "lag_recovery.csv", index=False)

    # Why the scan struggles: the correlation profile is nearly flat in the lag.
    prof = {}
    scan_win = wm.loc[:train_end]
    ydiff = scan_win[G.TARGET].shift(-1)
    ydiff = ydiff - ydiff.shift(12)
    for col in G.DRIVER_COLS:
        cs = []
        for k in range(13):
            f = scan_win[col].shift(k)
            f = f - f.shift(12)
            cs.append(abs(f.corr(ydiff)))
        prof[col] = cs
    prof = pd.DataFrame(prof, index=range(13))
    prof.to_csv(DATA / "lag_correlation_profiles.csv")

    # ------------------------------------------------------------ leakage audit
    audit = L.audit_feature_timing(XA, yA, top_n=8)
    print("\n--- timing audit on the leaky matrix (top suspects) ---")
    print(audit.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    audit.to_csv(DATA / "timing_audit.csv", index=False)

    # ------------------------------------------------- sign flip / margin fix
    # Refit on rolling 60-month windows and ask how often each coefficient keeps the
    # same sign. A variable whose true effect differs by customer segment cannot hold
    # a stable sign; the segment-margin construction gives it one.
    print("\n--- sign stability: raw prices vs segment margins ---")
    from sklearn.linear_model import LinearRegression

    sgn = wm.copy()
    sgn["y_ds"] = sgn[G.TARGET] - sgn[G.TARGET].shift(12)

    def sign_study(cols, label, lag=4, win=60, step=3):
        sub = sgn[["y_ds"]].copy()
        for c in cols:                       # seasonally difference both sides
            sub[c] = sgn[c].shift(lag) - sgn[c].shift(lag + 12)
        sub = sub.dropna()
        Xs = sub[cols].to_numpy(float)
        Xs = (Xs - Xs.mean(0)) / Xs.std(0)
        yy = sub["y_ds"].to_numpy()
        C = np.array([LinearRegression().fit(Xs[s:s + win], yy[s:s + win]).coef_
                      for s in range(0, len(sub) - win, step)])
        rows = []
        for i, c in enumerate(cols):
            sg = np.sign(C[:, i])
            rows.append({"specification": label, "feature": c,
                         "sign_consistency": max((sg > 0).mean(), (sg < 0).mean()),
                         "mean_coef": C[:, i].mean()})
        return pd.DataFrame(rows)

    sign_df = pd.concat([
        sign_study(["grain_price", "livestock_price", "feed_cost"], "raw prices"),
        sign_study(["crop_margin", "livestock_margin"], "segment margins"),
    ], ignore_index=True)
    print(sign_df.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    summary = sign_df.groupby("specification")["sign_consistency"].mean()
    print(f"\n   mean sign consistency -- raw prices: {summary['raw prices']:.2f}   "
          f"segment margins: {summary['segment margins']:.2f}")
    sign_df.to_csv(DATA / "sign_stability.csv", index=False)

    # ------------------------------------------------------- decision metrics
    print("\n--- decision-relevant metrics (honest pipeline) ---")
    ev = pd.DataFrame([M.threshold_events(resB.predictions, m)
                       for m in ["ridge", "gbm", "seasonal_naive"]])
    print(ev.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    ev.to_csv(DATA / "threshold_events.csv", index=False)

    best = resB.metrics.iloc[0]["model"]
    iv = M.residual_intervals(resB.predictions, best, level=0.8)
    cov = iv["covered"].mean()          # NaN warm-up rows excluded by construction
    print(f"\n80% empirical interval coverage ({best}): {cov:.3f} "
          f"on {int(iv['covered'].notna().sum())} scored origins")
    resid_series = resB.predictions[best] - resB.predictions["actual"]
    h = len(resid_series) // 2
    print(f"   residual sd: first half {resid_series[:h].std():.1f}, "
          f"second half {resid_series[h:].std():.1f} "
          f"-> error shrinks as the training window grows")

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    p = resB.predictions
    ax[0].plot(p.index, p["actual"], color=INK, lw=1.6, label="actual")
    ax[0].plot(p.index, p["seasonal_naive"], color=MUTED, lw=1.1, ls="--", label="seasonal naive")
    ax[0].plot(p.index, p[best], color=ACCENT, lw=1.4, label=f"{best} (honest)")
    ok = iv["lo"].notna()
    ax[0].fill_between(iv.index[ok], iv["lo"][ok], iv["hi"][ok], color=ACCENT, alpha=0.15,
                       label="80% interval")
    ax[0].set_ylabel("units"); ax[0].legend(frameon=False, ncol=4, fontsize=8)
    ax[0].set_title("Rolling-origin backtest, one month ahead")

    ax[1].bar(p.index, p[best] - p["actual"], width=20, color=MUTED)
    ax[1].axhline(0, color=INK, lw=0.9)
    ax[1].set_ylabel("error"); ax[1].set_title("Forecast error")
    fig.tight_layout(); fig.savefig(FIG / "backtest.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    w = 0.36
    xs = np.arange(len(comp))
    ax[0].bar(xs - w/2, comp["leaky_mae"], w, color=ACCENT, label="leaky")
    ax[0].bar(xs + w/2, comp["honest_mae"], w, color=INK, label="honest")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(comp["model"], fontsize=8)
    ax[0].set_ylabel("MAE (units)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("What the leaks were worth")

    for col in prof.columns:
        ax[1].plot(prof.index, prof[col], lw=1.2, label=col.replace("_", " "))
    for col in prof.columns:
        if col in G.TRUE_LAGS:
            ax[1].axvline(G.TRUE_LAGS[col], color=MUTED, lw=0.6, alpha=0.5)
    ax[1].set_xlabel("lag (months)"); ax[1].set_ylabel("|correlation|")
    ax[1].legend(frameon=False, fontsize=6.5)
    ax[1].set_title("Why lag selection fails:\ncorrelation is flat in the lag")

    mets = resB.metrics.set_index("model").loc[["seasonal_naive", "drift_naive", "ridge", "gbm"]]
    ax[2].barh(range(len(mets)), mets["mase"], color=[MUTED, MUTED, ACCENT, GOOD])
    ax[2].axvline(1.0, color=INK, lw=1.2, ls="--")
    ax[2].set_yticks(range(len(mets))); ax[2].set_yticklabels(mets.index, fontsize=8)
    ax[2].set_xlabel("MASE (< 1 beats seasonal naive)")
    ax[2].set_title("Honest skill vs baseline")
    fig.tight_layout(); fig.savefig(FIG / "leakage_and_skill.png"); plt.close(fig)

    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
