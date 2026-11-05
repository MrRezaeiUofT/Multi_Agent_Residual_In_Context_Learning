"""Synthetic-benchmark figure (Sec. 4.2): planted vs. recovered structure and accuracy vs. refinement.

Reads the per-seed results written by run_synthetic.py (or agent_step.py) and
re-evaluates the retained correction formulas with the released sandbox (maricl_core). Nothing
is fitted here; all numbers come from the stored runs.

Run (from this folder):
    python make_synthetic_figure.py
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = HERE
sys.path.insert(0, EXP)
from claude_backend import install_import_shims  # noqa: E402

install_import_shims()
import maricl_core as mc  # noqa: E402
from run_synthetic import FEATURES, sigmoid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(EXP, "results_agent"))
ap.add_argument("--pattern", default="seed[01].json")
ap.add_argument("--out", default=os.path.join(HERE, "synthetic_recovery"))
args = ap.parse_args()

runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.results, args.pattern)))]
assert runs, f"no results in {args.results}"
P_MIN = 0.1


def correction_raw(run, X):
    """Aggregated correction (ungated, Eq. 1) in raw y units for grid inputs X."""
    lo, hi = run["target_range"]
    active = [c for c in run["corrections"] if c["f_k"] and c["p_k"] > P_MIN]
    if not active:
        return np.zeros(len(X))
    p = np.array([c["p_k"] for c in active])
    deltas = np.stack([mc.execute_formula(c["f_k"], X, FEATURES, output_range=(-1.0, 1.0))
                       for c in active], axis=1)
    return (deltas @ (p / p.sum())) * (hi - lo)


def grid_inputs(**fixed):
    """Rows over the varied features; all other features 0."""
    keys = list(fixed)
    mesh = np.meshgrid(*[fixed[k] for k in keys], indexing="xy")
    X = np.zeros((mesh[0].size, len(FEATURES)))
    for k, m in zip(keys, mesh):
        X[:, FEATURES.index(k)] = m.ravel()
    return X, mesh[0].shape


# ---- representative seed: median headline (gated) test R^2 ----------------------------
r2_g = np.array([r["maricl_gated"]["r2"] for r in runs])
rep = runs[int(np.argsort(r2_g)[len(runs) // 2])]

# ---- (a,b) interaction surfaces over (x2, x5) ------------------------------------------
g = np.linspace(-2.5, 2.5, 121)
Xg, shape = grid_inputs(x2=g, x5=g)
planted_s = (2.0 * sigmoid(3.0 * Xg[:, 1] * Xg[:, 4])).reshape(shape)
planted_s -= planted_s.mean()
agree = []
for r in runs:
    rec = correction_raw(r, Xg).reshape(shape)
    rec -= rec.mean()
    agree.append(np.corrcoef(rec.ravel(), planted_s.ravel())[0, 1])
rec_s = correction_raw(rep, Xg).reshape(shape)
rec_s -= rec_s.mean()

# ---- (c) saturation along x4 -------------------------------------------------------------
u = np.linspace(-3, 3, 241)
X4, _ = grid_inputs(x4=u)
planted_4 = 1.2 * u**2 / (1 + u**2)
rec_4 = []
for r in runs:
    c = correction_raw(r, X4)
    rec_4.append(c - c[len(u) // 2])          # anchor at x4 = 0

# ---- (d) budget curves ----------------------------------------------------------------
T = np.array([b["T"] for b in runs[0]["budget_curve"]])
gated = np.array([[b["gated"] for b in r["budget_curve"]] for r in runs])
ungated = np.array([[b["ungated"] for b in r["budget_curve"]] for r in runs])
ref = {k: np.mean([r[k]["r2"] for r in runs]) for k in ("linear", "xgboost", "oracle")}

# ---- style ------------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, TRUTH = "#2a78d6", "#eb6834", "#1baf7a", "#3a3936"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#184f95", "#6da7ec", "#f0efec", "#ec8a87", "#a83232"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(6.3, 1.65))
outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 2.6], wspace=0.42)
left = outer[0].subgridspec(1, 2, wspace=0.08)
right = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42)
vmax = np.abs(planted_s).max()
ext = [g[0], g[-1], g[0], g[-1]]

ax = fig.add_subplot(left[0, 0])
ax.imshow(planted_s, origin="lower", extent=ext, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
ax.set_title("(a) Planted", loc="left")
ax.set_xlabel("$x_2$"); ax.set_ylabel("$x_5$")
ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])

ax = fig.add_subplot(left[0, 1])
im = ax.imshow(rec_s, origin="lower", extent=ext, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
ax.set_title("(b) MARICL", loc="left")
ax.set_xlabel("$x_2$"); ax.set_yticks([])
ax.set_xticks([-2, 0, 2])
ax.text(0.04, 0.04, rf"$r = {agree[runs.index(rep)]:.2f}$", transform=ax.transAxes, fontsize=7.5,
        color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
cax = ax.inset_axes([1.05, 0.0, 0.06, 1.0])
cb = fig.colorbar(im, cax=cax)
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.set_ticks([-0.5, 0, 0.5])

ax = fig.add_subplot(right[0, 0])
for i, c in enumerate(rec_4):
    ax.plot(u, c, color=ORANGE, lw=1.0, alpha=0.75, label="recovered" if i == 0 else None)
ax.plot(u, planted_4, color=TRUTH, lw=2, ls=(0, (4, 2)), label="planted")
ax.set_title(r"(c) Saturation in $x_4$", loc="left")
ax.set_xlabel("$x_4$"); ax.set_ylabel("effect on $y$")
ax.set_ylim(-0.05, 1.5)
ax.legend(loc="upper center", frameon=False, handlelength=1.8, borderaxespad=0.1)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = fig.add_subplot(right[0, 1])
for key, color, ls, name in (("oracle", TRUTH, (0, (1, 1.5)), "oracle"),
                             ("xgboost", AQUA, (0, (5, 2)), "XGBoost"),
                             ("linear", BLUE, (0, (5, 2)), "linear base")):
    ax.axhline(ref[key], color=color, lw=1.2, ls=ls)
    ax.text(T[-1], ref[key] + 0.012, f"{name} {ref[key]:.3f}", color=color, fontsize=7, va="bottom", ha="right")
for arr, alpha, name, ls in ((ungated, 0.55, "ungated", (0, (3, 1.5))),
                            (gated, 1.0, "MARICL", "-")):
    m = arr.mean(0)
    name = f"{name} {m[-1]:.3f}"
    ax.fill_between(T, arr.min(0), arr.max(0), color=ORANGE, alpha=0.15 * alpha, lw=0)
    ax.plot(T, m, color=ORANGE, lw=2, alpha=alpha, ls=ls, marker="o", ms=3, label=name)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.29), frameon=False, handlelength=1.5, borderaxespad=0.1)
ax.set_title("(d) Test $R^2$ vs. refinement", loc="left")
ax.set_xlabel("refinement iterations"); ax.set_ylabel("test $R^2$")
ax.set_xticks(T[::2])
ax.set_ylim(0.2, 1.04); ax.set_xlim(-0.3, T[-1] + 0.3)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.savefig(args.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(args.out + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)

summary = {
    "n_seeds": len(runs), "representative_seed": rep["seed"],
    "interaction_shape_r_all": [round(a, 4) for a in agree],
    **{f"{k}_r2": [round(r[k]["r2"], 4) for r in runs] for k in ("linear", "xgboost", "maricl_gated", "maricl_ungated", "oracle")},
}
print(json.dumps(summary, indent=1))
