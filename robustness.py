import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

YEARS = 20
MAXLAGS = 5
SHOCK_B = 'dlnvix_orth'
PERIODS = [
    ('2006-01-01', '2012-12-31', 'P1: 2006-12'),
    ('2013-01-01', '2019-12-31', 'P2: 2013-19'),
    ('2020-01-01', '2026-12-31', 'P3: 2020-26')
]

panel = pd.read_csv('panel.csv', parse_dates=['datetime']).set_index('datetime')
vix = (
    pd.read_csv(f'VIX_{YEARS}yr_data.csv', parse_dates=['datetime'])
    .set_index('datetime')['close']
)

dlnvix = np.log(vix / vix.shift(1))
panel['dlnvix'] = dlnvix.reindex(panel.index)
panel = panel.dropna()

stock_cols = [
    c for c in panel.columns
    if c.endswith('_ret') and c != 'spx_ret'
]

Xm = sm.add_constant(panel['spx_ret'])
orth_fit = sm.OLS(panel['dlnvix'], Xm).fit()
panel['dlnvix_orth'] = orth_fit.resid

print("=== Log-change shock: orthogonalization checks ===")
print("corr(dlnvix_orth, spx_ret):", panel['dlnvix_orth'].corr(panel['spx_ret']))
print("slope on spx_ret:", orth_fit.params['spx_ret'])
print("R-squared:", orth_fit.rsquared)

y = panel['JPM_ret']
b_raw = sm.OLS(
    y, sm.add_constant(panel[['spx_ret', 'dvix']])
).fit().params['dvix']

b_orth = sm.OLS(
    y, sm.add_constant(panel[['spx_ret', 'dvix_orth']])
).fit().params['dvix_orth']

print("\n=== FWL check (JPM) ===")
print(f"raw: {b_raw:.8f}   orth: {b_orth:.8f}")

rows = []

for col in stock_cols:
    y = panel[col]
    row = {'stock': col.replace('_ret', '')}

    for shock, tag in [('dvix_orth', 'level'), ('dlnvix_orth', 'log')]:
        X = sm.add_constant(panel[['spx_ret', shock]])
        fit = sm.OLS(y, X).fit(
            cov_type='HAC',
            cov_kwds={'maxlags': MAXLAGS}
        )

        row[f'{tag}_1sd_bps'] = fit.params[shock] * panel[shock].std() * 1e4
        row[f'{tag}_t'] = fit.tvalues[shock]

    rows.append(row)

partA = pd.DataFrame(rows).sort_values('log_1sd_bps')

print("\n=== PART A: Effect of a 1-SD pure volatility shock (bps) ===")
print(partA.round(2).to_string(index=False))
partA.to_csv('robust_A_level_vs_log.csv', index=False)


def period_dummies(index):
    return {
        label: ((index >= start) & (index <= end)).astype(float)
        for start, end, label in PERIODS
    }


D = period_dummies(panel.index)
design = pd.DataFrame(index=panel.index)

for label, d in D.items():
    design[f'const {label}'] = d
    design[f'mkt {label}'] = d * panel['spx_ret']
    design[f'vol {label}'] = d * panel[SHOCK_B]

vol_names = [f'vol {label}' for _, _, label in PERIODS]

rows = []

for col in stock_cols:
    fit = sm.OLS(panel[col], design).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': MAXLAGS}
    )

    b = fit.params[vol_names].values
    V = fit.cov_params().loc[vol_names, vol_names].values

    R = np.array([
        [-1, 1, 0],
        [-1, 0, 1]
    ])

    diff = R @ b
    W = diff @ np.linalg.inv(R @ V @ R.T) @ diff
    p = stats.chi2.sf(W, df=2)

    row = {'stock': col.replace('_ret', '')}

    for name in vol_names:
        row[f'{name[4:]} bps'] = fit.params[name] * panel[SHOCK_B].std() * 1e4
        row[f'{name[4:]} t'] = fit.tvalues[name]

    row['Wald p'] = p
    rows.append(row)

partB = pd.DataFrame(rows).sort_values('Wald p')

print(f"\n=== PART B: Volatility beta by subperiod ({SHOCK_B}) ===")
print(partB.round(3).to_string(index=False))
partB.to_csv('robust_B_subperiods.csv', index=False)

panel.to_csv('panel.csv')

print(
    "\nSaved robust_A_level_vs_log.csv, "
    "robust_B_subperiods.csv, and panel.csv"
)