# rolling_betas.py
# -----------------------------------------------------------------------------
# STEP FOUR: parameter instability. Step 3 gave each stock ONE vol beta for the
# whole 20 years. That single number assumes the sensitivity never changes.
# Here we re-run the SAME regression on a sliding 1-year window:
#
#     r_i,t = a + b_w * spx_ret_t + d_w * dvix_orth_t + e_t     (for each window w)
#
# and plot d_w (the vol beta) through time. If the line wanders far outside its
# own confidence band, the "constant beta" assumption is wrong for that stock.
# -----------------------------------------------------------------------------

import pandas as pd                                   # read the panel, hold the results
import numpy as np                                    # square roots and number helpers
import matplotlib.pyplot as plt                       # the plotting library
import statsmodels.api as sm                          # add_constant, same as before
from statsmodels.regression.rolling import RollingOLS # runs one OLS per window, fast

WINDOW = 252                                          # window length in trading days (about 1 year)

# --- load the panel (must already contain dvix_orth from step 2) -------------
panel = pd.read_csv('panel.csv', parse_dates=['datetime'])  # read the table; dates become real dates
panel = panel.set_index('datetime')                          # date is the row key again

stock_cols = [c for c in panel.columns                       # every column name...
              if c.endswith('_ret') and c != 'spx_ret']      # ...that is a stock return (not the market)

X = sm.add_constant(panel[['spx_ret', 'dvix_orth']])         # same right-hand side as step 3: const + market + vol shock

# --- run the rolling regression for each stock ------------------------------
beta_path = {}                                               # dict: stock -> Series of rolling vol betas
se_path = {}                                                 # dict: stock -> Series of their standard errors
full_beta = {}                                               # dict: stock -> the single full-sample vol beta (step 3's number)

for col in stock_cols:                                       # loop over the 9 stocks
    y = panel[col]                                           # this stock's daily return
    roll = RollingOLS(y, X, window=WINDOW).fit(cov_type='HCCM')  # one regression per 252-day window; HCCM = heteroskedasticity-robust SEs
    name = col.replace('_ret', '')                           # clean label, e.g. 'JPM'
    beta_path[name] = roll.params['dvix_orth']               # vol beta for every window, dated at the window's LAST day
    se_path[name] = roll.bse['dvix_orth']                    # standard error of that vol beta, same dates
    full_beta[name] = sm.OLS(y, X).fit().params['dvix_orth'] # one regression on all data, to draw as a reference line

betas = pd.DataFrame(beta_path)                              # table: rows = dates, columns = stocks
ses = pd.DataFrame(se_path)                                  # matching table of standard errors
betas.to_csv('rolling_vol_betas.csv')                        # save the numbers so the plot is reproducible

# --- the plot: one small panel per stock, all on the SAME y-scale -----------
# Same y-scale matters: it lets you compare how much each stock's beta moves.
# Order panels by full-sample vol beta, most negative first, like the step 3 table.
order = sorted(full_beta, key=full_beta.get)                 # stock names sorted by their full-sample vol beta

crises = [('2008-09-01', '2009-06-30', 'GFC'),               # stress periods to shade: start, end, label
          ('2020-02-15', '2020-06-30', 'COVID'),
          ('2022-01-01', '2022-10-31', '2022')]

line_color = '#2a6fdb'                                       # one color for every rolling beta line (same thing in every panel)

fig, axes = plt.subplots(3, 3, figsize=(13, 9),              # a 3 x 3 grid of small charts
                         sharex=True, sharey=True)           # shared axes so panels are directly comparable

for ax, name in zip(axes.flat, order):                       # walk the grid and the sorted stocks together
    b = betas[name].dropna()                                 # rolling betas (the first 251 days have no full window, so drop them)
    s = ses[name].reindex(b.index)                           # standard errors lined up to the same dates

    for start, end, label in crises:                         # shade each crisis window lightly in gray
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='0.9', zorder=0)

    ax.fill_between(b.index, b - 2 * s, b + 2 * s,           # confidence band: estimate +/- 2 standard errors
                    color=line_color, alpha=0.18, linewidth=0)
    ax.plot(b.index, b, color=line_color, linewidth=1.5)     # the rolling vol beta itself
    ax.axhline(full_beta[name], color='0.35',                # the single full-sample number from step 3...
               linestyle='--', linewidth=1)                  # ...dashed, so you can see how far the rolling one strays
    ax.axhline(0, color='black', linewidth=0.6)              # zero line: above it the stock RISES on vol shocks

    ax.set_title(f"{name}   full-sample = {full_beta[name]:+.4f}", fontsize=10)  # stock name + its step 3 number
    ax.grid(axis='y', color='0.92', linewidth=0.6)           # faint horizontal gridlines only
    for side in ('top', 'right'):                            # remove the top and right box edges (less clutter)
        ax.spines[side].set_visible(False)

for ax in axes[:, 0]:                                        # only the left column gets a y-axis label
    ax.set_ylabel('vol beta (return per\n1-pt VIX shock)', fontsize=9)

# one legend for the whole figure, explaining each visual element once
from matplotlib.patches import Patch                         # for legend boxes (band, shading)
from matplotlib.lines import Line2D                          # for legend lines
handles = [Line2D([], [], color=line_color, linewidth=1.5, label=f'rolling {WINDOW}-day vol beta'),
           Patch(color=line_color, alpha=0.18, label='+/- 2 robust standard errors'),
           Line2D([], [], color='0.35', linestyle='--', label='full-sample vol beta (step 3)'),
           Patch(color='0.9', label='stress periods: GFC, COVID, 2022')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9)

fig.suptitle('Rolling 1-year sensitivity to pure volatility shocks (orthogonalized dVIX)', fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])                    # leave room at the bottom for the legend and top for the title
fig.savefig('rolling_vol_betas.png', dpi=150)                # write the image file to your folder
print("Saved rolling_vol_betas.png and rolling_vol_betas.csv")
plt.show()                                                   # also pop the chart up on screen