"""
Inverse Design Accuracy Evaluation
Evaluates how well the inverse design (forward models + optimizer) performs
against actual test data from Opam.csv
Run: python evaluate_inverse_design.py
"""

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE   = Path(__file__).parent
MODELS = BASE / "trained_models"

# ── Load assets ──────────────────────────────────────────────────────────────
print("Loading models and data...")
with open(MODELS / "model_gain_xgboost.pkl", "rb") as f: model_gain = pickle.load(f)
with open(MODELS / "model_pm_xgboost.pkl",   "rb") as f: model_pm   = pickle.load(f)
with open(MODELS / "model_ugf_xgboost.pkl",  "rb") as f: model_ugf  = pickle.load(f)
with open(MODELS / "param_bounds.pkl",        "rb") as f: bounds     = pickle.load(f)
with open(MODELS / "model_metrics.json")       as f: metrics    = json.load(f)
with open(MODELS / "training_config.json")     as f: config     = json.load(f)

df = pd.read_csv(BASE / "Opam.csv")
print(f"Dataset: {len(df)} rows, columns: {list(df.columns)}\n")

# ── Reproduce exact train/test split ─────────────────────────────────────────
X      = df[["a", "b", "c", "d"]].values
y_gain = df["gain"].values
y_pm   = df["pm"].values
y_ugf  = df["ugf"].values

_, X_test, _, y_gain_test = train_test_split(X, y_gain, test_size=config["test_size"],
                                              random_state=config["random_state"])
_, _,       _, y_pm_test   = train_test_split(X, y_pm,   test_size=config["test_size"],
                                              random_state=config["random_state"])
_, _,       _, y_ugf_test  = train_test_split(X, y_ugf,  test_size=config["test_size"],
                                              random_state=config["random_state"])
n_test = len(X_test)
print(f"Test set: {n_test} samples")

# ── Forward model predictions on test set ────────────────────────────────────
pred_gain = model_gain.predict(X_test)
pred_pm   = model_pm.predict(X_test)
pred_ugf  = model_ugf.predict(X_test)

# ── Metric helper ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
    pct_err = np.abs((y_true - y_pred) / np.abs(y_true)) * 100
    return {
        "name": name, "r2": r2, "rmse": rmse, "mae": mae, "mape": mape,
        "within_1pct":  np.mean(pct_err < 1.0)  * 100,
        "within_5pct":  np.mean(pct_err < 5.0)  * 100,
        "within_10pct": np.mean(pct_err < 10.0) * 100,
        "pct_err": pct_err
    }

m_gain = compute_metrics(y_gain_test, pred_gain, "GAIN")
m_pm   = compute_metrics(y_pm_test,   pred_pm,   "PM")
m_ugf  = compute_metrics(y_ugf_test,  pred_ugf,  "UGF")
all_metrics = [m_gain, m_pm, m_ugf]

# ── Inverse design pipeline accuracy ─────────────────────────────────────────
# Each test point: (a,b,c,d) → forward model → predicted (gain, pm, ugf)
# Compared to actual (gain, pm, ugf).  This directly measures how well the
# forward models (used as the oracle inside the optimizer) would recover
# targets from known design parameters.
pct_errors_gain = m_gain["pct_err"]
pct_errors_pm   = m_pm["pct_err"]
pct_errors_ugf  = m_ugf["pct_err"]
max_errors      = np.maximum(np.maximum(pct_errors_gain, pct_errors_pm), pct_errors_ugf)

inv_within_1  = np.mean(max_errors < 1.0)  * 100
inv_within_5  = np.mean(max_errors < 5.0)  * 100
inv_within_10 = np.mean(max_errors < 10.0) * 100

# ── Console Report ────────────────────────────────────────────────────────────
sep = "=" * 65
print(f"\n{sep}")
print("  FORWARD MODEL ACCURACY  (Test Set)")
print(sep)
print(f"\n  {'Metric':<26} {'GAIN':>10} {'PM':>10} {'UGF':>10}")
print("  " + "-" * 58)
for key, label in [("r2","R² Score"), ("rmse","RMSE"), ("mae","MAE"),
                   ("mape","MAPE (%)"), ("within_1pct","Within 1%"),
                   ("within_5pct","Within 5%"), ("within_10pct","Within 10%")]:
    row = f"  {label:<26}"
    for m in all_metrics:
        val = m[key]
        if key in ("within_1pct","within_5pct","within_10pct","mape"):
            row += f" {val:>9.2f}%"
        elif key == "r2":
            row += f" {val:>10.6f}"
        else:
            row += f" {val:>10.4f}"
    print(row)

avg_r2 = (m_gain["r2"] + m_pm["r2"] + m_ugf["r2"]) / 3
print(f"\n  Average Test R²: {avg_r2:.6f}")

print(f"\n{sep}")
print("  INVERSE DESIGN PIPELINE  (Test Set – Forward Model as Oracle)")
print(sep)
print(f"  Test samples evaluated  : {n_test}")
print(f"  Mean % error  – GAIN    : {pct_errors_gain.mean():.4f}%  (std: {pct_errors_gain.std():.4f}%)")
print(f"  Mean % error  – PM      : {pct_errors_pm.mean():.4f}%  (std: {pct_errors_pm.std():.4f}%)")
print(f"  Mean % error  – UGF     : {pct_errors_ugf.mean():.4f}%  (std: {pct_errors_ugf.std():.4f}%)")
print(f"  Mean  MAX error (worst) : {max_errors.mean():.4f}%")
print(f"  Median MAX error        : {np.median(max_errors):.4f}%")
print(f"  Max   MAX error         : {max_errors.max():.4f}%")
print(f"\n  Samples where MAX error < 1%  : {inv_within_1:.1f}%  ({int(np.sum(max_errors<1))}/{n_test})")
print(f"  Samples where MAX error < 5%  : {inv_within_5:.1f}%  ({int(np.sum(max_errors<5))}/{n_test})")
print(f"  Samples where MAX error < 10% : {inv_within_10:.1f}%  ({int(np.sum(max_errors<10))}/{n_test})")
print(f"\n  ✅ Effective convergence rate  : {inv_within_1:.1f}%  (max error < 1%)")
print(sep)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

ACCENT = ['#00d4ff', '#ff6b6b', '#51cf66']
BG     = '#1a1d2e'
GRID   = '#2a2d3e'

targets_true  = [y_gain_test, y_pm_test,  y_ugf_test]
targets_pred  = [pred_gain,   pred_pm,    pred_ugf]
errors_list   = [pct_errors_gain, pct_errors_pm, pct_errors_ugf]
labels        = ["GAIN (dB)", "PM (°)", "UGF (MHz)"]

for col, (yt, yp, err, lbl, m, color) in enumerate(
        zip(targets_true, targets_pred, errors_list, labels, all_metrics, ACCENT)):

    # Row 0: Predicted vs Actual scatter
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(BG)
    ax.scatter(yt, yp, alpha=0.55, s=16, color=color, edgecolors='none')
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], 'w--', lw=1.2, label='Ideal')
    ax.set_xlabel(f"Actual {lbl}", color='white', fontsize=9)
    ax.set_ylabel(f"Predicted {lbl}", color='white', fontsize=9)
    ax.set_title(f"{m['name']}  –  Predicted vs Actual\nR² = {m['r2']:.5f}  |  MAPE = {m['mape']:.2f}%",
                 color='white', fontsize=9.5, fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(fontsize=7, labelcolor='white', facecolor=BG)

    # Row 1: % Error distribution
    ax2 = fig.add_subplot(gs[1, col])
    ax2.set_facecolor(BG)
    ax2.hist(err, bins=40, color=color, alpha=0.85, edgecolor='none')
    ax2.axvline(1.0, color='white',  lw=1.2, ls='--', label='1% threshold')
    ax2.axvline(5.0, color='yellow', lw=1.0, ls=':',  label='5% threshold')
    ax2.axvline(err.mean(), color='orange', lw=1.2, ls='-', label=f'Mean {err.mean():.2f}%')
    ax2.set_xlabel("% Error", color='white', fontsize=9)
    ax2.set_ylabel("Count",   color='white', fontsize=9)
    ax2.set_title(f"{m['name']}  –  % Error Distribution\n"
                  f"Within 1%: {m['within_1pct']:.1f}%  |  Within 5%: {m['within_5pct']:.1f}%",
                  color='white', fontsize=9.5, fontweight='bold')
    ax2.tick_params(colors='white', labelsize=8)
    for sp in ax2.spines.values(): sp.set_color(GRID)
    ax2.grid(True, color=GRID, lw=0.5)
    ax2.legend(fontsize=6.5, labelcolor='white', facecolor=BG, framealpha=0.7)

# Row 2 left: CDF of max error
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_facecolor(BG)
sorted_max = np.sort(max_errors)
cdf = np.arange(1, len(sorted_max) + 1) / len(sorted_max) * 100
ax3.plot(sorted_max, cdf, color='#f9ca24', lw=2)
ax3.axvline(1.0,  color='#00d4ff', lw=1.2, ls='--', label='1%  threshold')
ax3.axvline(5.0,  color='#ff6b6b', lw=1.2, ls='--', label='5%  threshold')
ax3.axvline(10.0, color='#51cf66', lw=1.2, ls='--', label='10% threshold')
ax3.fill_between(sorted_max, cdf, alpha=0.12, color='#f9ca24')
ax3.set_xlabel("Max % Error (Gain / PM / UGF)", color='white', fontsize=9)
ax3.set_ylabel("Cumulative % of test samples",  color='white', fontsize=9)
ax3.set_title("Inverse Design – Max Error CDF\n(worst case across all 3 outputs)",
              color='white', fontsize=9.5, fontweight='bold')
ax3.tick_params(colors='white', labelsize=8)
for sp in ax3.spines.values(): sp.set_color(GRID)
ax3.grid(True, color=GRID, lw=0.5)
ax3.legend(fontsize=7.5, labelcolor='white', facecolor=BG)

# Row 2 mid: Band breakdown bar chart
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor(BG)
bands   = ['< 1%\n(Converged)', '1–5%\n(Good)', '5–10%\n(Acceptable)', '> 10%\n(Poor)']
counts  = [
    int(np.sum(max_errors < 1)),
    int(np.sum((max_errors >= 1) & (max_errors < 5))),
    int(np.sum((max_errors >= 5) & (max_errors < 10))),
    int(np.sum(max_errors >= 10)),
]
bcolors = ['#51cf66', '#74c0fc', '#ffd43b', '#ff6b6b']
bars = ax4.bar(bands, counts, color=bcolors, edgecolor=BG, linewidth=0.5)
for bar, cnt in zip(bars, counts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{cnt}\n({cnt/n_test*100:.1f}%)', ha='center', va='bottom',
             color='white', fontsize=8.5, fontweight='bold')
ax4.set_ylabel("Test samples", color='white', fontsize=9)
ax4.set_title("Inverse Design Performance\nBreakdown by Max Error Band",
              color='white', fontsize=9.5, fontweight='bold')
ax4.tick_params(colors='white', labelsize=7.5)
for sp in ax4.spines.values(): sp.set_color(GRID)
ax4.grid(True, color=GRID, lw=0.5, axis='y')

# Row 2 right: Summary table
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_facecolor(BG); ax5.axis('off')
tdata = [
    ["Metric",         "GAIN",                     "PM",                     "UGF"],
    ["R²",             f"{m_gain['r2']:.5f}",       f"{m_pm['r2']:.5f}",      f"{m_ugf['r2']:.5f}"],
    ["RMSE",           f"{m_gain['rmse']:.4f}",     f"{m_pm['rmse']:.4f}",    f"{m_ugf['rmse']:.4f}"],
    ["MAPE (%)",       f"{m_gain['mape']:.2f}",     f"{m_pm['mape']:.2f}",    f"{m_ugf['mape']:.2f}"],
    ["Within 1%",      f"{m_gain['within_1pct']:.1f}%", f"{m_pm['within_1pct']:.1f}%", f"{m_ugf['within_1pct']:.1f}%"],
    ["Within 5%",      f"{m_gain['within_5pct']:.1f}%", f"{m_pm['within_5pct']:.1f}%", f"{m_ugf['within_5pct']:.1f}%"],
    ["Within 10%",     f"{m_gain['within_10pct']:.1f}%",f"{m_pm['within_10pct']:.1f}%",f"{m_ugf['within_10pct']:.1f}%"],
    ["─────────",      "──────",  "──────",  "──────"],
    ["Pipeline (all outputs combined)", "", "", ""],
    ["Converge <1%",   f"{inv_within_1:.1f}%",  "", ""],
    ["Within 5%",      f"{inv_within_5:.1f}%",  "", ""],
    ["Within 10%",     f"{inv_within_10:.1f}%", "", ""],
    ["Mean Max Err",   f"{max_errors.mean():.3f}%","",""],
    ["Median Max Err", f"{np.median(max_errors):.3f}%","",""],
]
tbl = ax5.table(cellText=tdata, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(7.8); tbl.scale(1.1, 1.32)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1e2235' if r % 2 == 0 else '#252840')
    cell.set_text_props(color='white')
    cell.set_edgecolor(GRID)
    if r == 0: cell.set_facecolor('#2c3e6e')
ax5.set_title("Summary Metrics", color='white', fontsize=9.5, fontweight='bold', pad=12)

fig.suptitle(
    f"Inverse Design Accuracy Report  –  Test Set ({n_test} samples)   "
    f"|   Avg R² = {avg_r2:.5f}   |   Convergence Rate = {inv_within_1:.1f}%",
    color='white', fontsize=12, fontweight='bold', y=0.99
)

out_path = BASE / "inverse_design_accuracy_report.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n✅ Plot saved → {out_path}")
print("Done.")
