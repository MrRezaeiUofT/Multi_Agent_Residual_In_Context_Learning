"""One figure for both examples: what Claude's corrections recover, and how accuracy improves.

Reads the stored results in results/regression and results/classification and re-evaluates the
retained correction formulas with the released sandbox (maricl_core). The only model fitted here
is the logistic base of the classification example, refit exactly as in the run.

Top row, regression: (a) the true x2*x5 term, (b) Claude's correction on the same grid,
(c) the effect of x4, (d) test R^2 from the base model through refinement.
Bottom row, classification on a slice at x4 = 1 (x1 = x3 = x6 = 0): (e) true classes,
(f) the logistic base, (g) after Claude's correction, (h) test accuracy by refinement iteration.

Run (from this folder):
    python make_figure.py
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claude_backend import install_import_shims  # noqa: E402

install_import_shims()
import classification as cl  # noqa: E402
import maricl_core as mc  # noqa: E402
from maricl_headline import maricl_predict_proba  # noqa: E402
from regression import FEATURES, sigmoid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layout", choices=["wide", "vertical"], default="wide",
                help="wide: 2 x 4 for the README; vertical: 4 x 2, one column wide, for the paper")
ap.add_argument("--out", default=None)
args = ap.parse_args()
args.out = args.out or os.path.join(HERE, "figures", "synthetic_results" + ("_vertical" if args.layout == "vertical" else ""))


def load(task):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "results", task, "seed[01].json")))]
    assert runs, f"no results for {task}"
    return runs


def grid(**fixed):
    """Rows over a 2-D (x2, x5) grid, or a 1-D x4 line; every other feature at its fixed value."""
    X = np.zeros((next(iter(fixed.values())).size, len(FEATURES)))
    for k, v in fixed.items():
        X[:, FEATURES.index(k)] = v
    return X


# ---------------------------------------------------------------- regression
reg = load("regression")


def correction_raw(run, X):
    """Aggregated correction (ungated, Eq. 1) in raw y units."""
    lo, hi = run["target_range"]
    active = [c for c in run["corrections"] if c["f_k"] and c["p_k"] > 0.1]
    p = np.array([c["p_k"] for c in active])
    deltas = np.stack([mc.execute_formula(c["f_k"], X, FEATURES, output_range=(-1.0, 1.0)) for c in active], 1)
    return (deltas @ (p / p.sum())) * (hi - lo)


g = np.linspace(-2.5, 2.5, 121)
gx, gy = np.meshgrid(g, g, indexing="xy")
Xs = grid(x2=gx.ravel(), x5=gy.ravel())
true_s = (2.0 * sigmoid(3.0 * gx * gy))
true_s -= true_s.mean()
rep_r = reg[int(np.argsort([r["maricl_gated"]["r2"] for r in reg])[len(reg) // 2])]
rec_s = correction_raw(rep_r, Xs).reshape(gx.shape)
rec_s -= rec_s.mean()
r_shape = np.corrcoef(rec_s.ravel(), true_s.ravel())[0, 1]

u = np.linspace(-3, 3, 241)
true_4 = 1.2 * u**2 / (1 + u**2)
rec_4 = [(lambda c: c - c[len(u) // 2])(correction_raw(r, grid(x4=u))) for r in reg]

T_r = np.array([b["T"] for b in reg[0]["budget_curve"]])
# each curve starts at the base model (drawn at x = -2), then the first formula (0) and refinement (1..T)
curve_r = np.array([[r["linear"]["r2"]] + [b["gated"] for b in r["budget_curve"]] for r in reg])
ref_r = {k: float(np.mean([r[k]["r2"] for r in reg])) for k in ("linear", "xgboost", "oracle")}

# ---------------------------------------------------------------- classification
clf = load("classification")
rep_c = clf[int(np.argsort([r["maricl"]["accuracy"] for r in clf])[len(clf) // 2])]
X, y, _, sp = cl.make_data(rep_c["seed"])
base = LogisticRegression(max_iter=1000).fit(X[sp["train"]], y[sp["train"]])
model = mc.MARICL(task="classification")
model.n_classes, model.feature_names = cl.C, FEATURES
chosen = [mc.Correction(T_k=c["T_k"], f_k=c["f_k"]) if c["f_k"] else None for c in rep_c["corrections"]]
gc = np.linspace(-2.5, 2.5, 241)
cx, cy = np.meshgrid(gc, gc, indexing="xy")
Xc = grid(x2=cx.ravel(), x5=cy.ravel(), x4=np.ones(cx.size))
Pc = base.predict_proba(Xc)
classes = [cl.true_scores(Xc).argmax(1), Pc.argmax(1),
           maricl_predict_proba(model, Xc, Pc, chosen, [c["p_k"] for c in rep_c["corrections"]],
                                rep_c["beta"]).argmax(1)]

T_c = np.array([b["T"] for b in clf[0]["budget_curve"]])
curve_c = np.array([[r["logistic"]["accuracy"]] + [b["accuracy"] for b in r["budget_curve"]] for r in clf])
ref_c = {k: float(np.mean([r[k]["accuracy"] for r in clf])) for k in ("logistic", "xgboost", "oracle")}

# ---------------------------------------------------------------- style
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, TRUTH = "#2a78d6", "#eb6834", "#1baf7a", "#3a3936"
CLASSES = ["#6d5bd0", "#d0507c", "#d9a521"]        # validated with the dataviz palette checker
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#184f95", "#6da7ec", "#f0efec", "#ec8a87", "#a83232"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def heat(ax, data, cmap, title, ylabel, **kw):
    ax.imshow(data, origin="lower", extent=[-2.5, 2.5, -2.5, 2.5], cmap=cmap, interpolation="nearest", **kw)
    ax.set_title(title, loc="left")
    ax.set_xlabel("$x_2$")
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    if ylabel:
        ax.set_ylabel("$x_5$")
    else:
        ax.set_yticklabels([])


def clean(ax, ygrid_only=True):
    ax.grid(axis="y" if ygrid_only else "both", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def curve(ax, T, arr, refs, title, ylabel, ylim):
    for (name, val), color, ls in zip(refs, (TRUTH, AQUA, BLUE), ((0, (1, 1.5)), (0, (5, 2)), (0, (5, 2)))):
        ax.axhline(val, color=color, lw=1.2, ls=ls, label=name)
    T = np.concatenate([[-2], T])
    m = arr.mean(0)
    ax.fill_between(T, arr.min(0), arr.max(0), color=ORANGE, alpha=0.15, lw=0)
    ax.plot(T, m, color=ORANGE, lw=2, marker="o", ms=2.6, label="MARICL")
    ax.set_title(title, loc="left")
    ax.set_xlabel("refinement iteration")
    ticks = [-2] + list(range(0, int(T[-1]) + 1, 2))
    ax.set_xticks(ticks, ["base"] + [str(t) for t in ticks[1:]])
    ax.set_ylim(*ylim); ax.set_xlim(-2.5, T[-1] + 0.3)
    clean(ax)


VERTICAL = args.layout == "vertical"
if VERTICAL:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5})
    fig, grid = plt.subplots(4, 2, figsize=(3.3, 5.5), gridspec_kw=dict(wspace=0.32, hspace=0.78))
    # columns are tasks; rows are truth, Claude's result, a second view, and performance
    P = {k: grid[r, c] for k, (r, c) in dict(a=(0, 0), b=(1, 0), c=(2, 0), d=(3, 0),
                                              e=(0, 1), f=(1, 1), g=(2, 1), h=(3, 1)).items()}
    clf_panels = (("e", 0, "(e) True classes"), ("f", 2, "(f) After Claude's fix"), ("g", 1, "(g) Logistic base"))
else:
    fig, grid = plt.subplots(2, 4, figsize=(6.9, 3.55), gridspec_kw=dict(wspace=0.3, hspace=0.62))
    P = {k: grid[r, c] for k, (r, c) in dict(a=(0, 0), b=(0, 1), c=(0, 2), d=(0, 3),
                                              e=(1, 0), f=(1, 1), g=(1, 2), h=(1, 3)).items()}
    clf_panels = (("e", 0, "(e) True classes"), ("f", 1, "(f) Logistic base"), ("g", 2, "(g) After Claude's fix"))
LEFT = {"a", "e"} if not VERTICAL else {"a", "b", "e", "f", "g"}

vmax = np.abs(true_s).max()
heat(P["a"], true_s, DIVERGING, r"(a) True $x_2 x_5$ term", True, vmin=-vmax, vmax=vmax)
heat(P["b"], rec_s, DIVERGING, "(b) Claude's correction", "b" in LEFT, vmin=-vmax, vmax=vmax)
P["b"].text(0.05, 0.06, rf"$r = {r_shape:.2f}$", transform=P["b"].transAxes, fontsize=7 if VERTICAL else 7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax = P["c"]
for i, c in enumerate(rec_4):
    ax.plot(u, c, color=ORANGE, lw=1.0, alpha=0.8, label="Claude" if i == 0 else None)
ax.plot(u, true_4, color=TRUTH, lw=1.8, ls=(0, (4, 2)), label="true")
ax.set_title(r"(c) Effect of $x_4$ on $y$", loc="left")
ax.set_xlabel("$x_4$"); ax.set_ylim(-0.05, 1.5)
ax.legend(loc="upper center", frameon=False, handlelength=1.6, borderaxespad=0.1, ncol=2, columnspacing=1.0)
clean(ax, ygrid_only=False)

curve(P["d"], T_r, curve_r, [("best possible", ref_r["oracle"]), ("XGBoost", ref_r["xgboost"]),
                             ("base model", ref_r["linear"])], "(d) Test $R^2$", "test $R^2$", (0.1, 1.04))

cmap = ListedColormap(CLASSES)
for key, idx, title in clf_panels:
    heat(P[key], classes[idx].reshape(cx.shape), cmap, title, key in LEFT or key == "e",
         vmin=-0.5, vmax=cl.C - 0.5)
for txt, (px, py) in (("0", (-1.5, 1.5)), ("1", (1.5, 1.5)), ("2", (0.55, -0.45))):
    P["e"].text(px, py, txt, color="white", fontsize=8 if VERTICAL else 9, fontweight="bold", ha="center", va="center")

curve(P["h"], T_c, curve_c, [("best possible", ref_c["oracle"]), ("XGBoost", ref_c["xgboost"]),
                             ("base model", ref_c["logistic"])], "(h) Test accuracy", "test accuracy", (0.25, 1.04))

if VERTICAL:
    for key, label in (("a", "Regression"), ("e", "Classification")):
        pos = P[key].get_position()
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.045, label, ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")
else:
    for key, label in (("a", "Regression"), ("e", "Classification")):
        pos = P[key].get_position()
        fig.text(0.035, pos.y0 + pos.height / 2, label, rotation=90, va="center", ha="center",
                 fontsize=9, fontweight="bold")

h, l = P["h"].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.5, -0.07 if not VERTICAL else -0.035),
           ncol=4 if not VERTICAL else 2, frameon=False, handlelength=2.0, columnspacing=1.8,
           fontsize=8 if not VERTICAL else 7)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(args.out + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print(json.dumps({"regression_shape_r": round(float(r_shape), 4), "regression_seed": rep_r["seed"],
                  "classification_seed": rep_c["seed"], "ref_regression": ref_r, "ref_classification": ref_c}))
