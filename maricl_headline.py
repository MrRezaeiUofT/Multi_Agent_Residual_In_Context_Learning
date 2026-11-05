"""Headline configuration of MARICL (paper Secs. 3.4 and 3.5).

maricl_core.MARICL keeps the iterate with the lowest training loss, and its predict method
applies the ungated rule (Eq. 1) or a fixed blend (Eq. 2). The steps below add what the paper's
protocol selects on validation data. None of them calls an LLM, and all of them work on a fitted
MARICL model and leave maricl_core.py unchanged.

Regression:

    select          chooses the retained iterate m_k* on validation data, then fixes p_k on
                    the training split (Sec. 3.5)
    maricl_predict  applies the outer gate lambda(x) with the steeper confidence profile
                    (Eqs. 20 and 21)

    chosen, scores, _ = select(model, X_train, y_train, X_val, y_val,
                               base.predict(X_train), base.predict(X_val))
    y_hat = maricl_predict(model, X_test, base.predict(X_test), chosen, scores)

Classification:

    select_classification  chooses m_k* by validation log loss, then fixes p_k as macro F1 on
                           the training split (Sec. 3.5)
    select_beta            chooses the blend beta by validation log loss
    maricl_predict_proba   applies Eq. 2 with those choices

    chosen, scores, _ = select_classification(model, X_train, y_train, X_val, y_val,
                                              base.predict_proba(X_train), base.predict_proba(X_val))
    beta = select_beta(model, X_val, y_val, base.predict_proba(X_val), chosen, scores)
    proba = maricl_predict_proba(model, X_test, base.predict_proba(X_test), chosen, scores, beta)

The per-correction temperature tau_k of Eq. 14 is not implemented here or in maricl_core.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, mean_absolute_error


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _require(model, task):
    if model.task != task:
        raise ValueError(f"this function supports {task} only, not {model.task}")


def pool_distance(model, X):
    """d_k(x): distance to the high-residual pool, clipped to [0, 1] (Sec. 3.3)."""
    Xs = (X - model._mu) / model._sd
    dmin = np.linalg.norm(Xs[:, None, :] - model._X_hr_std[None, :, :], axis=-1).min(axis=1)
    return np.minimum(dmin / model._D95, 1.0)


def maricl_predict(model, X, base_pred, corrections, scores, gated=True):
    """Eq. 21 when gated, otherwise Eq. 1. Returns base_pred if no correction clears p_min."""
    _require(model, "regression")
    active = [(c, p) for c, p in zip(corrections, scores) if c is not None and p > model.p_min]
    if not active:
        return base_pred
    deltas = np.stack([model._apply_single(c, X, model.feature_names) for c, _ in active], axis=1)
    p = np.array([p for _, p in active])
    corr = deltas @ (p / p.sum())            # shared c(x) cancels in alpha_k (Eq. 19)
    if not gated:
        return base_pred + corr
    c = _sigmoid(model.gamma * (1.0 - 2.0 * pool_distance(model, X)))   # steeper profile
    lam = np.minimum(p.max() * c, 1.0)                                  # Eq. 20
    return base_pred + lam * corr


def select(model, X_tr, y_tr, X_va, y_va, b_tr, b_va, t_max=None):
    """Per correction: argmin validation MAE over iterates t <= t_max; p_k on train.

    Returns (corrections, scores, retained_t). A correction with no usable iterate comes back
    as None with a score of 0.0, which maricl_predict skips.
    """
    _require(model, "regression")
    features = model.feature_names
    chosen, scores, ts = [], [], []
    for s in model.states_:
        best = None
        for e in s.history:
            if t_max is not None and e["t"] > t_max:
                continue
            try:
                mae = mean_absolute_error(y_va, b_va + model._apply_single(e["correction"], X_va, features))
            except Exception:
                continue
            if best is None or mae < best[0] - 1e-12:
                best = (mae, e)
        if best is None:
            chosen.append(None); scores.append(0.0); ts.append(None); continue
        corr = best[1]["correction"]
        mae_tr = mean_absolute_error(y_tr, b_tr + model._apply_single(corr, X_tr, features))
        chosen.append(corr); scores.append(float(np.exp(-mae_tr / model.tau_reg))); ts.append(best[1]["t"])
    return chosen, scores, ts


def maricl_predict_proba(model, X, base_proba, corrections, scores, beta=None):
    """Eq. 2. Returns base_proba if no correction clears p_min; beta defaults to model.beta."""
    _require(model, "classification")
    active = [(c, p) for c, p in zip(corrections, scores) if c is not None and p > model.p_min]
    if not active:
        return base_proba
    beta = model.beta if beta is None else beta
    logits = np.log(np.clip(base_proba, 1e-9, 1.0))
    Q = []
    for c, _ in active:
        s = logits + model._apply_single(c, X, model.feature_names)
        s = s - s.max(axis=1, keepdims=True)
        q = np.exp(s)
        Q.append(q / q.sum(axis=1, keepdims=True))
    p = np.array([p for _, p in active])
    mix = (np.stack(Q, axis=1) * (p / p.sum())[None, :, None]).sum(axis=1)   # shared c(x) cancels
    P = beta * base_proba + (1.0 - beta) * mix
    return P / P.sum(axis=1, keepdims=True)


def select_classification(model, X_tr, y_tr, X_va, y_va, P_tr, P_va, t_max=None):
    """Per correction: argmin validation log loss over iterates t <= t_max; p_k is train macro F1.

    Returns (corrections, scores, retained_t). A correction with no usable iterate comes back
    as None with a score of 0.0, which maricl_predict_proba skips.
    """
    _require(model, "classification")
    features = model.feature_names
    chosen, scores, ts = [], [], []
    for s in model.states_:
        best = None
        for e in s.history:
            if t_max is not None and e["t"] > t_max:
                continue
            loss, _ = model._correction_loss(e["correction"], X_va, y_va, P_va, features)
            if not np.isfinite(loss):
                continue
            if best is None or loss < best[0] - 1e-12:
                best = (loss, e)
        if best is None:
            chosen.append(None); scores.append(0.0); ts.append(None); continue
        corr = best[1]["correction"]
        _, proba = model._correction_loss(corr, X_tr, y_tr, P_tr, features)
        chosen.append(corr); scores.append(float(f1_score(y_tr, proba.argmax(axis=1), average="macro")))
        ts.append(best[1]["t"])
    return chosen, scores, ts


def select_beta(model, X_va, y_va, P_va, corrections, scores, grid=(0.3, 0.5, 0.7)):
    """Choose the blend beta from the paper's grid by validation log loss."""
    _require(model, "classification")
    labels = list(range(model.n_classes))
    losses = [log_loss(y_va, maricl_predict_proba(model, X_va, P_va, corrections, scores, b), labels=labels)
              for b in grid]
    return float(grid[int(np.argmin(losses))])
