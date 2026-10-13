import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

MAXLAGS = 5
SHOCK = "dlnvix_orth"
PERIODS = [
    ("2006-01-01", "2012-12-31", "P1"),
    ("2013-01-01", "2019-12-31", "P2"),
    ("2020-01-01", "2026-12-31", "P3"),
]

panel = pd.read_csv("panel.csv", parse_dates=["datetime"]).set_index("datetime")
stock_cols = [c for c in panel.columns if c.endswith("_ret") and c != "spx_ret"]
SIGMA = panel[SHOCK].std()

lo, hi = panel[SHOCK].quantile([0.01, 0.99])
p_winsor = panel.copy()
p_winsor[SHOCK] = panel[SHOCK].clip(lower=lo, upper=hi)

top10 = panel[SHOCK].abs().nlargest(10).index
p_drop10 = panel.drop(index=top10)

mkt_cut = panel["spx_ret"].abs().quantile(0.99)
p_dropmkt = panel[panel["spx_ret"].abs() <= mkt_cut]

specs = {
    "base": panel,
    "winsor": p_winsor,
    "drop10": p_drop10,
    "dropmkt": p_dropmkt,
}

print("=== what got trimmed ===")
print(
    f"winsor: shock clipped to [{lo:+.4f}, {hi:+.4f}]  ({(panel[SHOCK] < lo).sum() + (panel[SHOCK] > hi).sum()} days capped)"
)
print("drop10: removed these days (date: shock):")
for d in top10:
    print(f"    {d.date()}: {panel.loc[d, SHOCK]:+.4f}")
print(
    f"dropmkt: removed {len(panel) - len(p_dropmkt)} days with |spx_ret| > {mkt_cut:.4f}"
)


def full_sample(df, col):
    X = sm.add_constant(df[["spx_ret", SHOCK]])
    fit = sm.OLS(df[col], X).fit(cov_type="HAC", cov_kwds={"maxlags": MAXLAGS})
    return fit.params[SHOCK] * SIGMA * 1e4, fit.tvalues[SHOCK]


rows = []
for col in stock_cols:
    row = {"stock": col.replace("_ret", "")}
    for name, df in specs.items():
        bps, t = full_sample(df, col)
        row[f"{name} bps"] = bps
        row[f"{name} t"] = t
    rows.append(row)

tabA = pd.DataFrame(rows).sort_values("base bps")
print("\n=== PART A: 1-SD log-shock effect (bps) and t-stat, by spec ===")
print(tabA.round(2).to_string(index=False))
tabA.to_csv("robust_outliers_A.csv", index=False)


def subperiod_test(df, col):
    design = pd.DataFrame(index=df.index)
    for start, end, label in PERIODS:
        d = ((df.index >= start) & (df.index <= end)).astype(float)
        design[f"const {label}"] = d
        design[f"mkt {label}"] = d * df["spx_ret"]
        design[f"vol {label}"] = d * df[SHOCK]
    fit = sm.OLS(df[col], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": MAXLAGS}
    )
    names = [f"vol {label}" for _, _, label in PERIODS]
    b = fit.params[names].values
    V = fit.cov_params().loc[names, names].values
    R = np.array([[-1, 1, 0], [-1, 0, 1]])
    diff = R @ b
    W = diff @ np.linalg.inv(R @ V @ R.T) @ diff
    out = {
        f"{lab} bps": fit.params[f"vol {lab}"] * SIGMA * 1e4
        for _, _, lab in PERIODS
    }
    out["Wald p"] = stats.chi2.sf(W, df=2)
    return out


all_B = []
for name, df in specs.items():
    rows = [
        {"stock": c.replace("_ret", ""), **subperiod_test(df, c)}
        for c in stock_cols
    ]
    tabB = pd.DataFrame(rows).sort_values("Wald p")
    print(f"\n=== PART B ({name}): subperiod 1-SD effects (bps) + Wald p ===")
    print(tabB.round(3).to_string(index=False))
    all_B.append(tabB.assign(spec=name))

pd.concat(all_B).to_csv("robust_outliers_B.csv", index=False)
print("\nSaved robust_outliers_A.csv and robust_outliers_B.csv")
print(f"Bonferroni cutoff for 9 Wald tests: {0.05/9:.4f}")