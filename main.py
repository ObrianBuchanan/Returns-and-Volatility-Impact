import os # 

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.regression.rolling import RollingOLS

STOCKS = ["AAPL", "MSFT", "NVDA", "GOOGL", "WMT", "JPM", "XOM", "KO", "JNJ"]
YEARS = 20
MAXLAGS = 5
WINDOW = 252
OUT = "results"
PERIODS = [("2006-01-01", "2012-12-31", "P1"),
           ("2013-01-01", "2019-12-31", "P2"),
           ("2020-01-01", "2026-12-31", "P3")]
CRISES = [("2008-09-01", "2009-06-30"),
          ("2020-02-15", "2020-06-30"),
          ("2022-01-01", "2022-10-31")]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def raw_file(symbol):
    return pd.read_csv(f"{symbol}_{YEARS}yr_data.csv", parse_dates=["datetime"]).set_index("datetime")


def stock_cols(panel):
    return [c for c in panel.columns if c.endswith("_ret") and c != "spx_ret"]


def hac_fit(y, X):
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": MAXLAGS})


def build_panel():
    section("1. BUILD PANEL")
    spx = raw_file("SPX")["return"].rename("spx_ret")
    vix = raw_file("VIX")["close"]
    dvix = vix.diff().rename("dvix")
    dlnvix = np.log(vix / vix.shift(1)).rename("dlnvix")
    stocks = [raw_file(s)["return"].rename(f"{s}_ret") for s in STOCKS]
    panel = pd.concat([spx, dvix, dlnvix] + stocks, axis=1, join="inner").dropna()
    print("shape:", panel.shape)
    print("date range:", panel.index.min().date(), "to", panel.index.max().date())
    print("any NaNs:", panel.isna().any().any())
    return panel


def diagnose(panel):
    section("2. DIAGNOSTICS")
    for sym in ["SPX", "VIX"] + STOCKS:
        df = raw_file(sym)
        print(f"{sym:6s} rows={len(df):5d}  {df.index.min().date()} to {df.index.max().date()}")
    print("\nrows per year:")
    print(panel.groupby(panel.index.year).size().to_dict())
    rets = [c for c in panel.columns if c.endswith("_ret")]
    summary = pd.DataFrame({"ann_mean_%": panel[rets].mean() * 252 * 100,
                            "ann_vol_%": panel[rets].std() * np.sqrt(252) * 100})
    print("\n" + summary.round(2).to_string())
    print("\nlargest daily moves:")
    for c in rets:
        top = panel[c].abs().nlargest(3).index
        print(f"{c:10s} " + " | ".join(f"{d.date()}: {panel.loc[d, c]:+.3f}" for d in top))
    print("\ncorrelation with SPX at t-1 / t / t+1:")
    for c in stock_cols(panel):
        lag0 = panel[c].corr(panel["spx_ret"])
        prev = panel[c].corr(panel["spx_ret"].shift(1))
        nxt = panel[c].corr(panel["spx_ret"].shift(-1))
        flag = "  <-- MISALIGNED?" if max(abs(prev), abs(nxt)) > abs(lag0) else ""
        print(f"{c:10s} {prev:+.3f}  {lag0:+.3f}  {nxt:+.3f}{flag}")


def orthogonalize(panel):
    section("3. ORTHOGONALIZE VOL SHOCKS")
    X = sm.add_constant(panel["spx_ret"])
    fits = {}
    for raw, orth in [("dvix", "dvix_orth"), ("dlnvix", "dlnvix_orth")]:
        fit = sm.OLS(panel[raw], X).fit()
        panel[orth] = fit.resid
        fits[raw] = fit
        print(f"{orth:12s} corr={panel[orth].corr(panel['spx_ret']):.2e}  "
              f"slope={fit.params['spx_ret']:+.3f}  R2={fit.rsquared:.3f}")

    y = panel["JPM_ret"]
    b_raw = sm.OLS(y, sm.add_constant(panel[["spx_ret", "dvix"]])).fit().params["dvix"]
    b_orth = sm.OLS(y, sm.add_constant(panel[["spx_ret", "dvix_orth"]])).fit().params["dvix_orth"]
    print(f"FWL check (JPM): raw {b_raw:.8f}  orth {b_orth:.8f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fit = fits["dvix"]
    axes[0].scatter(panel["spx_ret"], panel["dvix"], s=6, alpha=0.4)
    xs = np.linspace(panel["spx_ret"].min(), panel["spx_ret"].max(), 100)
    axes[0].plot(xs, fit.params["const"] + fit.params["spx_ret"] * xs, color="red")
    axes[0].set(title=f"Leverage effect (slope = {fit.params['spx_ret']:.2f})", xlabel="SPX return", ylabel="dVIX")
    axes[1].plot(panel.index, panel["dlnvix_orth"], linewidth=0.5)
    axes[1].axhline(0, color="red", linewidth=0.8)
    axes[1].set(title="dlnvix_orth over time")
    fig.tight_layout()
    fig.savefig(f"{OUT}/orthogonalization.png", dpi=150)
    plt.close(fig)
    return panel


def per_stock_betas(panel):
    section("4. FULL-SAMPLE BETAS (level vs log shock, 1-SD effect in bps)")
    rows = []
    for col in stock_cols(panel):
        row = {"stock": col.replace("_ret", "")}
        for shock, tag in [("dvix_orth", "level"), ("dlnvix_orth", "log")]:
            fit = hac_fit(panel[col], sm.add_constant(panel[["spx_ret", shock]]))
            if tag == "log":
                row["alpha"] = fit.params["const"]
                row["mkt_beta"] = fit.params["spx_ret"]
                row["r2"] = fit.rsquared
            row[f"{tag}_bps"] = fit.params[shock] * panel[shock].std() * 1e4
            row[f"{tag}_t"] = fit.tvalues[shock]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("log_bps")
    print(table.round(4).to_string(index=False))
    table.to_csv(f"{OUT}/full_sample_betas.csv", index=False)


def rolling_betas(panel, shock):
    X = sm.add_constant(panel[["spx_ret", shock]])
    sd = panel[shock].std() * 1e4
    betas, ses, full = {}, {}, {}
    for col in stock_cols(panel):
        name = col.replace("_ret", "")
        roll = RollingOLS(panel[col], X, window=WINDOW).fit(cov_type="HCCM")
        betas[name] = roll.params[shock] * sd
        ses[name] = roll.bse[shock] * sd
        full[name] = sm.OLS(panel[col], X).fit().params[shock] * sd
    betas, ses = pd.DataFrame(betas), pd.DataFrame(ses)
    betas.to_csv(f"{OUT}/rolling_betas_{shock}.csv")

    color = "#2a6fdb"
    order = sorted(full, key=full.get)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, order):
        b = betas[name].dropna()
        s = ses[name].reindex(b.index)
        for start, end in CRISES:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="0.9", zorder=0)
        ax.fill_between(b.index, b - 2 * s, b + 2 * s, color=color, alpha=0.18, linewidth=0)
        ax.plot(b.index, b, color=color, linewidth=1.5)
        ax.axhline(full[name], color="0.35", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{name}   full-sample = {full[name]:+.1f} bps", fontsize=10)
        ax.grid(axis="y", color="0.92", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("bps per 1-SD shock", fontsize=9)
    handles = [Line2D([], [], color=color, linewidth=1.5, label=f"rolling {WINDOW}-day effect"),
               Patch(color=color, alpha=0.18, label="+/- 2 robust SE"),
               Line2D([], [], color="0.35", linestyle="--", label="full-sample effect"),
               Patch(color="0.9", label="stress periods: GFC, COVID, 2022")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Rolling 1-year effect of a 1-SD pure volatility shock ({shock})", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f"{OUT}/rolling_betas_{shock}.png", dpi=150)
    plt.close(fig)


def subperiod_test(df, col, shock, sigma):
    design = pd.DataFrame(index=df.index)
    for start, end, label in PERIODS:
        d = ((df.index >= start) & (df.index <= end)).astype(float)
        design[f"const {label}"] = d
        design[f"mkt {label}"] = d * df["spx_ret"]
        design[f"vol {label}"] = d * df[shock]
    fit = hac_fit(df[col], design)
    names = [f"vol {label}" for _, _, label in PERIODS]
    b = fit.params[names].values
    V = fit.cov_params().loc[names, names].values
    R = np.array([[-1, 1, 0], [-1, 0, 1]])
    diff = R @ b
    W = diff @ np.linalg.inv(R @ V @ R.T) @ diff
    out = {f"{label} bps": fit.params[f"vol {label}"] * sigma * 1e4 for _, _, label in PERIODS}
    out.update({f"{label} t": fit.tvalues[f"vol {label}"] for _, _, label in PERIODS})
    out["Wald p"] = stats.chi2.sf(W, df=2)
    return out


def robustness(panel, shock="dlnvix_orth"):
    section("6. OUTLIER ROBUSTNESS + SUBPERIOD STABILITY (log shock)")
    sigma = panel[shock].std()
    lo, hi = panel[shock].quantile([0.01, 0.99])
    winsor = panel.copy()
    winsor[shock] = panel[shock].clip(lower=lo, upper=hi)
    top10 = panel[shock].abs().nlargest(10).index
    mkt_cut = panel["spx_ret"].abs().quantile(0.99)
    specs = {"base": panel,
             "winsor": winsor,
             "drop10": panel.drop(index=top10),
             "dropmkt": panel[panel["spx_ret"].abs() <= mkt_cut]}

    print(f"winsor bounds: [{lo:+.4f}, {hi:+.4f}]")
    print("drop10 days: " + ", ".join(f"{d.date()} ({panel.loc[d, shock]:+.3f})" for d in top10))
    print(f"dropmkt: removed {len(panel) - len(specs['dropmkt'])} days with |spx_ret| > {mkt_cut:.4f}")

    rows = []
    for col in stock_cols(panel):
        row = {"stock": col.replace("_ret", "")}
        for name, df in specs.items():
            fit = hac_fit(df[col], sm.add_constant(df[["spx_ret", shock]]))
            row[f"{name} bps"] = fit.params[shock] * sigma * 1e4
            row[f"{name} t"] = fit.tvalues[shock]
        rows.append(row)
    tabA = pd.DataFrame(rows).sort_values("base bps")
    print("\nfull-sample 1-SD effect by spec:")
    print(tabA.round(2).to_string(index=False))
    tabA.to_csv(f"{OUT}/robust_full_sample.csv", index=False)

    all_B = []
    for name, df in specs.items():
        tabB = pd.DataFrame([{"stock": c.replace("_ret", ""), **subperiod_test(df, c, shock, sigma)}
                             for c in stock_cols(panel)]).sort_values("Wald p")
        print(f"\nsubperiods ({name}):")
        print(tabB.round(3).to_string(index=False))
        all_B.append(tabB.assign(spec=name))
    pd.concat(all_B).to_csv(f"{OUT}/robust_subperiods.csv", index=False)
    print(f"\nBonferroni cutoff (9 tests): {0.05 / 9:.4f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = build_panel()
    diagnose(panel)
    panel = orthogonalize(panel)
    panel.to_csv("panel.csv")
    per_stock_betas(panel)
    section("5. ROLLING BETAS")
    for shock in ["dvix_orth", "dlnvix_orth"]:
        rolling_betas(panel, shock)
        print(f"saved {OUT}/rolling_betas_{shock}.png")
    robustness(panel)
    section("DONE")
    print(f"all tables and charts are in ./{OUT}/")


if __name__ == "__main__":
    main()