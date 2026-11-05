"""Synthetic planted-ground-truth benchmark for MARICL with a Claude backbone.

    y = 0.4 x1 - 0.3 x3 + 2 sigmoid(3 x2 x5) + 1.2 x4^2 / (1 + x4^2) + eps,
    x ~ N(0, I_6),  eps ~ N(0, 0.1^2),  N = 1000 (600 / 200 / 200),  x6 irrelevant.

Protocol (paper Sec. 3): residuals, high-residual pool, critiques and p_k on train; the retained
iterate m_k* is chosen on validation; test is used only for final metrics. Predictions are
reported for the gated headline configuration (Eq. 21) and the ungated variant (Eq. 1).

Usage:
    python run_synthetic.py --seed 0                 # real run (needs Anthropic credentials)
    python run_synthetic.py --seed 0 --mock --T 3    # offline pipeline check, no API calls
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from claude_backend import ClaudeLLM, install_import_shims

install_import_shims()
import maricl_core as mc  # noqa: E402  (released MARICL implementation, unmodified)

FEATURES = [f"x{i}" for i in range(1, 7)]
FEATURE_DESCS = ["standardised input feature (semantics unknown)"] * len(FEATURES)
DOMAIN = ("Synthetic regression benchmark with standardised inputs. Feature semantics are "
          "unknown and no domain knowledge applies.")
N, D = 1000, 6


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def planted(X):
    x1, x2, x3, x4, x5 = (X[:, i] for i in range(5))
    return 0.4 * x1 - 0.3 * x3 + 2.0 * sigmoid(3.0 * x2 * x5) + 1.2 * x4**2 / (1.0 + x4**2)


def make_data(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    f = planted(X)
    y = f + rng.normal(0.0, 0.1, N)
    lo, hi = float(y.min()), float(y.max())
    idx = np.random.RandomState(seed).permutation(N)
    split = {"train": idx[:600], "val": idx[600:800], "test": idx[800:]}
    return X, (y - lo) / (hi - lo), (f - lo) / (hi - lo), split, (lo, hi)


# ---------------------------------------------------------------- recovery (syntax tree)
def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def term_flags(expr: str) -> dict:
    tree = ast.parse(expr, mode="eval")
    inter = spurious12 = sat4 = False
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            l, r = _names(n.left), _names(n.right)
            if ("x2" in l and "x5" in r) or ("x5" in l and "x2" in r):
                inter = True
            if ("x1" in l and "x2" in r) or ("x2" in l and "x1" in r):
                spurious12 = True
            if "x4" in l and "x4" in r:
                sat4 = True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and "x4" in _names(n.left):
            sat4 = True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and "x4" in _names(n.right):
            sat4 = True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id != "clip" \
                and "x4" in _names(n):
            sat4 = True
    used = sorted(_names(tree) & set(FEATURES))
    return {"interaction_x2x5": inter, "nonlinear_x4": sat4, "uses_x6": "x6" in used,
            "spurious_x1x2": spurious12, "features_used": used}


# ---------------------------------------------------------------- prediction
def pool_distance(model, X):
    Xs = (X - model._mu) / model._sd
    dmin = np.linalg.norm(Xs[:, None, :] - model._X_hr_std[None, :, :], axis=-1).min(axis=1)
    return np.minimum(dmin / model._D95, 1.0)


def maricl_predict(model, X, base_pred, corrections, scores, gated):
    active = [(c, p) for c, p in zip(corrections, scores) if c is not None and p > model.p_min]
    if not active:
        return base_pred
    deltas = np.stack([model._apply_single(c, X, model.feature_names) for c, _ in active], axis=1)
    p = np.array([p for _, p in active])
    corr = deltas @ (p / p.sum())            # shared c(x) cancels in alpha_k (Eq. 19)
    if not gated:
        return base_pred + corr
    c = sigmoid(model.gamma * (1.0 - 2.0 * pool_distance(model, X)))   # steeper profile
    lam = np.minimum(p.max() * c, 1.0)                                  # Eq. 20
    return base_pred + lam * corr


def select(model, X_tr, y_tr, X_va, y_va, b_tr, b_va, t_max):
    """Per correction: argmin validation MAE over iterates t <= t_max; p_k on train."""
    chosen, scores, ts = [], [], []
    for s in model.states_:
        best = None
        for e in s.history:
            if e["t"] > t_max:
                continue
            try:
                mae = mean_absolute_error(y_va, b_va + model._apply_single(e["correction"], X_va, FEATURES))
            except Exception:
                continue
            if best is None or mae < best[0] - 1e-12:
                best = (mae, e)
        if best is None:
            chosen.append(None); scores.append(0.0); ts.append(None); continue
        corr = best[1]["correction"]
        mae_tr = mean_absolute_error(y_tr, b_tr + model._apply_single(corr, X_tr, FEATURES))
        chosen.append(corr); scores.append(float(np.exp(-mae_tr / model.tau_reg))); ts.append(best[1]["t"])
    return chosen, scores, ts


# ---------------------------------------------------------------- offline stand-in
class _MockRunnable:
    def __init__(self, schema, rng):
        self.schema, self.rng = schema, rng

    def invoke(self, messages):
        name = self.schema.__name__
        if name == "Hypothesis":
            return self.schema(hypothesised_pattern="mock", implicated_features=["x2", "x5", "x4"],
                               functional_form_guess="a*sigmoid(b*x2*x5) + c*x4**2/(1+x4**2)",
                               rationale="offline mock")
        if name == "Correction":
            a, b, c = self.rng.uniform(0.05, 0.3), self.rng.uniform(0.0, 0.15), self.rng.uniform(0, 0.2)
            return self.schema(T_k="mock", f_k=f"{a:.3f}*sigmoid(3*x2*x5) + {b:.3f}*x4**2/(1+x4**2) - {c:.3f}")
        return self.schema(diagnosis="mock", structural_or_coefficient="coefficient", suggested_change="mock")


class MockLLM:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def with_structured_output(self, schema):
        return _MockRunnable(schema, self.rng)


# ---------------------------------------------------------------- main
def load_dotenv(path=".env"):
    """Read KEY=VALUE lines into the environment (existing variables win). Values are never printed."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def run_experiment(seed, T, llm, llm_creative, out_dir, tag, calls_path=None, fit_log=None):
    """Run one seed end to end; returns the results dict (also written to out_dir/tag.json)."""
    import contextlib
    X, ys, fs, sp, (lo, hi) = make_data(seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    lin = LinearRegression().fit(X[tr], ys[tr])
    b_tr, b_va, b_te = lin.predict(X[tr]), lin.predict(X[va]), lin.predict(X[te])
    mc.configure_llms(llm, llm_creative)
    model = mc.MARICL(task="regression", K=2, kappa=0.3, T=T, B=10, p_min=0.1,
                      gamma=2.0, tau_fail=0.1, tau_reg=0.2, verbose=True)
    t0 = time.time()
    sink = open(fit_log, "w") if fit_log else None
    try:
        with contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext():
            model.fit(X[tr], ys[tr], base_model=lin, feature_names=FEATURES,
                      feature_descs=FEATURE_DESCS, domain_context=DOMAIN, reg_target_range=(0.0, 1.0))
    finally:
        if sink:
            sink.close()
    fit_seconds = time.time() - t0
    xgb = XGBRegressor(random_state=seed).fit(X[tr], ys[tr])

    def metrics(pred, idx):
        return {"r2": float(r2_score(ys[idx], pred)), "mae": float(mean_absolute_error(ys[idx], pred))}

    chosen, scores, ts = select(model, X[tr], ys[tr], X[va], ys[va], b_tr, b_va, T)
    res = {
        "seed": seed, "T": T, "target_range": [lo, hi], "fit_seconds": round(fit_seconds, 1),
        "linear": metrics(b_te, te), "xgboost": metrics(xgb.predict(X[te]), te),
        "oracle": metrics(fs[te], te),
        "maricl_gated": metrics(maricl_predict(model, X[te], b_te, chosen, scores, True), te),
        "maricl_ungated": metrics(maricl_predict(model, X[te], b_te, chosen, scores, False), te),
        "corrections": [], "budget_curve": [],
    }
    for k, (s, c, p, t) in enumerate(zip(model.states_, chosen, scores, ts)):
        res["corrections"].append({
            "k": k + 1, "p_k": p, "retained_t": t,
            "hypothesis": s.hypothesis.hypothesised_pattern,
            "form_guess": s.hypothesis.functional_form_guess,
            "T_k": c.T_k if c else None, "f_k": c.f_k if c else None,
            "flags": term_flags(c.f_k) if c else None,
            "history": [{"t": e["t"], "f_k": e["correction"].f_k, "train_mae": e["loss"]}
                        for e in s.history],
        })
    for t_max in range(T + 1):
        ch, sc, _ = select(model, X[tr], ys[tr], X[va], ys[va], b_tr, b_va, t_max)
        res["budget_curve"].append({
            "T": t_max,
            "gated": metrics(maricl_predict(model, X[te], b_te, ch, sc, True), te)["r2"],
            "ungated": metrics(maricl_predict(model, X[te], b_te, ch, sc, False), te)["r2"],
        })
    if calls_path and os.path.exists(calls_path):
        calls = [json.loads(line) for line in open(calls_path)]
        res["calls"] = {"n": len(calls), "input_tokens": sum(c["input_tokens"] for c in calls),
                        "output_tokens": sum(c["output_tokens"] for c in calls)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = f"{'mock_' if args.mock else ''}seed{args.seed}"
    calls_path = os.path.join(args.out, f"calls_{tag}.jsonl")
    if os.path.exists(calls_path):
        os.remove(calls_path)
    if args.mock:
        llm = llm_creative = MockLLM(args.seed)
    else:
        llm = ClaudeLLM(args.model, args.effort, calls_path, role="base")
        llm_creative = ClaudeLLM(args.model, args.effort, calls_path, role="encoder")
    res = run_experiment(args.seed, args.T, llm, llm_creative, args.out, tag, calls_path)
    print(json.dumps({k: res[k] for k in ("linear", "xgboost", "maricl_gated", "maricl_ungated", "oracle")}))


if __name__ == "__main__":
    sys.exit(main())
