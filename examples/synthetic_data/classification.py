"""Classification example on synthetic data: Claude fixes a logistic model's residuals.

Each example belongs to the class with the largest score, with 2% label noise:

    s0 = -2 x2 x5
    s1 =  2 x2 x5
    s2 = 1.5 (x4^2 - 1) + 0.8 x1 - 0.6 x3 + 0.5

Here x ~ N(0, I_6), N = 1000 (600 / 200 / 200), and x6 is irrelevant. Logistic regression can use
the linear x1 and x3 terms, but it cannot represent the x2 x5 interaction or the even x4 term.

Protocol (paper Sec. 3.5): residuals, the high-residual pool, critiques and p_k use the training
split; the retained iterate m_k* and the blend beta are chosen on validation; test is used only
for final metrics. Predictions follow Eq. 2.

Usage:
    python classification.py --seed 0                 # real run (needs Anthropic credentials)
    python classification.py --seed 0 --mock --T 3    # offline pipeline check, no API calls
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

from claude_backend import ClaudeLLM, install_import_shims

install_import_shims()
import maricl_core as mc  # noqa: E402  (released MARICL implementation, unmodified)
from maricl_headline import maricl_predict_proba, select_beta, select_classification  # noqa: E402
from regression import FEATURE_DESCS, FEATURES, load_dotenv, term_flags  # noqa: E402

DOMAIN = ("Synthetic 3-class classification benchmark with standardised inputs. Feature semantics "
          "are unknown and no domain knowledge applies.")
N, D, C = 1000, 6, 3


def true_scores(X):
    x1, x2, x3, x4, x5 = (X[:, i] for i in range(5))
    return np.stack([-2.0 * x2 * x5, 2.0 * x2 * x5,
                     1.5 * (x4**2 - 1.0) + 0.8 * x1 - 0.6 * x3 + 0.5], axis=1)


def make_data(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    clean = true_scores(X).argmax(axis=1)
    y = clean.copy()
    flip = rng.random(N) < 0.02
    y[flip] = rng.integers(0, C, size=int(flip.sum()))
    idx = np.random.RandomState(seed).permutation(N)
    split = {"train": idx[:600], "val": idx[600:800], "test": idx[800:]}
    return X, y, clean, split


def metrics(y, proba):
    pred = proba.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "log_loss": float(log_loss(y, proba, labels=list(range(C))))}


def label_metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


def recovered_structure(model, correction, X):
    """How closely one correction's class scores track the two true nonlinear terms."""
    delta = model._apply_single(correction, X, FEATURES)
    x2, x4, x5 = X[:, 1], X[:, 3], X[:, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        inter = np.corrcoef(delta[:, 1] - delta[:, 0], x2 * x5)[0, 1]
        even = np.corrcoef(delta[:, 2] - 0.5 * (delta[:, 0] + delta[:, 1]), x4**2)[0, 1]
    return {"r_interaction_x2x5": float(np.nan_to_num(inter)), "r_even_x4": float(np.nan_to_num(even))}


# ---------------------------------------------------------------- offline stand-in
class _MockRunnable:
    def __init__(self, schema, rng):
        self.schema, self.rng = schema, rng

    def invoke(self, messages):
        name = self.schema.__name__
        if name == "Hypothesis":
            return self.schema(hypothesised_pattern="mock", implicated_features=["x2", "x5", "x4"],
                               functional_form_guess="[-a*x2*x5, a*x2*x5, b*(x4**2 - 1)]",
                               rationale="offline mock")
        if name == "Correction":
            a, b = self.rng.uniform(0.5, 2.0), self.rng.uniform(0.2, 1.5)
            return self.schema(T_k="mock", f_k=f"[-{a:.3f}*x2*x5, {a:.3f}*x2*x5, {b:.3f}*(x4**2 - 1)]")
        return self.schema(diagnosis="mock", structural_or_coefficient="coefficient", suggested_change="mock")


class MockLLM:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def with_structured_output(self, schema):
        return _MockRunnable(schema, self.rng)


# ---------------------------------------------------------------- main
def run_experiment(seed, T, llm, llm_creative, out_dir, tag, calls_path=None, fit_log=None):
    """Run one seed end to end; returns the results dict (also written to out_dir/tag.json)."""
    X, y, clean, sp = make_data(seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    base = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    P_tr, P_va, P_te = base.predict_proba(X[tr]), base.predict_proba(X[va]), base.predict_proba(X[te])
    mc.configure_llms(llm, llm_creative)
    model = mc.MARICL(task="classification", K=2, kappa=0.3, T=T, B=10, p_min=0.1,
                      gamma=2.0, tau_fail=0.5, verbose=True)
    t0 = time.time()
    sink = open(fit_log, "w") if fit_log else None
    try:
        with contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext():
            model.fit(X[tr], y[tr], base_model=base, feature_names=FEATURES,
                      feature_descs=FEATURE_DESCS, domain_context=DOMAIN, n_classes=C)
    finally:
        if sink:
            sink.close()
    fit_seconds = time.time() - t0
    xgb = XGBClassifier(random_state=seed).fit(X[tr], y[tr])

    def maricl(t_max):
        chosen, scores, ts = select_classification(model, X[tr], y[tr], X[va], y[va], P_tr, P_va, t_max)
        beta = select_beta(model, X[va], y[va], P_va, chosen, scores)
        return chosen, scores, ts, beta, maricl_predict_proba(model, X[te], P_te, chosen, scores, beta)

    chosen, scores, ts, beta, proba = maricl(T)
    res = {
        "seed": seed, "T": T, "fit_seconds": round(fit_seconds, 1), "class_counts": np.bincount(y).tolist(),
        "logistic": metrics(y[te], P_te), "xgboost": metrics(y[te], xgb.predict_proba(X[te])),
        "oracle": label_metrics(y[te], clean[te]), "maricl": metrics(y[te], proba), "beta": beta,
        "corrections": [], "budget_curve": [],
    }
    for k, (s, c, p, t) in enumerate(zip(model.states_, chosen, scores, ts)):
        res["corrections"].append({
            "k": k + 1, "p_k": p, "retained_t": t,
            "hypothesis": s.hypothesis.hypothesised_pattern,
            "form_guess": s.hypothesis.functional_form_guess,
            "T_k": c.T_k if c else None, "f_k": c.f_k if c else None,
            "flags": term_flags(c.f_k) if c else None,
            "structure": recovered_structure(model, c, X[te]) if c else None,
            "history": [{"t": e["t"], "f_k": e["correction"].f_k, "train_log_loss": e["loss"]}
                        for e in s.history],
        })
    for t_max in range(T + 1):
        res["budget_curve"].append({"T": t_max, **metrics(y[te], maricl(t_max)[4])})
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
    ap.add_argument("--out", default="results_api/classification")
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
    print(json.dumps({k: res[k] for k in ("logistic", "xgboost", "maricl", "oracle")}))


if __name__ == "__main__":
    sys.exit(main())
