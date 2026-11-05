"""Headline configuration of MARICL for regression (paper Secs. 3.4 and 3.5).

maricl_core.MARICL keeps the iterate with the lowest training loss, and its predict method
applies the ungated rule (Eq. 1). The paper's headline numbers add two steps, neither of which
calls an LLM:

    select          chooses the retained iterate m_k* on validation data, then fixes p_k on
                    the training split (Sec. 3.5)
    maricl_predict  applies the outer gate lambda(x) with the steeper confidence profile
                    (Eqs. 20 and 21)

Both work on a fitted MARICL model and leave maricl_core.py unchanged:

    chosen, scores, _ = select(model, X_train, y_train, X_val, y_val,
                               base.predict(X_train), base.predict(X_val))
    y_hat = maricl_predict(model, X_test, base.predict(X_test), chosen, scores)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _require_regression(model):
    if model.task != "regression":
        raise ValueError("maricl_headline supports regression only")


def pool_distance(model, X):
    """d_k(x): distance to the high-residual pool, clipped to [0, 1] (Sec. 3.3)."""
    Xs = (X - model._mu) / model._sd
    dmin = np.linalg.norm(Xs[:, None, :] - model._X_hr_std[None, :, :], axis=-1).min(axis=1)
    return np.minimum(dmin / model._D95, 1.0)


def maricl_predict(model, X, base_pred, corrections, scores, gated=True):
    """Eq. 21 when gated, otherwise Eq. 1. Returns base_pred if no correction clears p_min."""
    _require_regression(model)
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
    _require_regression(model)
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
