"""Training-only Spearman/BH screening followed by nested Lasso.

The candidate family contains every supplied lncRNA, including untestable columns
(p=1). Correlations use observed pairs, never imputed expression or outcomes.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.stats import rankdata, t as student_t, false_discovery_control
from sklearn.linear_model import Lasso, lasso_path
from sklearn.exceptions import ConvergenceWarning

try:
    import celer
except ImportError:
    celer = None
SOLVER = "celer-"+celer.__version__ if celer is not None else "sklearn-coordinate-descent"
ALPHA_RATIOS = np.logspace(0, -3, 25)


def solve_path(x, y, **kwargs):
    if celer is None:
        return lasso_path(x, y, **kwargs)
    return celer.celer_path(x, y, "lasso", alphas=kwargs["alphas"], tol=kwargs["tol"],
            max_iter=100, max_epochs=kwargs["max_iter"], return_n_iter=True, verbose=0)



def screen(x, y, *, min_abs_rho=.20, max_fdr=.05, min_pairs=10):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Patient axes disagree")
    if not 0 <= min_abs_rho <= 1 or not 0 < max_fdr <= 1 or min_pairs < 3:
        raise ValueError("Invalid correlation screen")
    rho = np.zeros((x.shape[1], y.shape[1]), dtype=np.float32)
    p = np.ones(rho.shape, dtype=np.float64)
    n = np.zeros(rho.shape, dtype=np.int32)
    yg = {}
    for j in range(y.shape[1]):
        yg.setdefault(np.packbits(np.isfinite(y[:, j])).tobytes(), []).append(j)
    # Global X missingness groups remain valid after any Y-row restriction.
    # Build once, instead of repacking every gene for every target mask.
    xg = {}
    for k in range(x.shape[1]):
        xg.setdefault(np.packbits(np.isfinite(x[:, k])).tobytes(), []).append(k)
    for js in yg.values():
        rows = np.isfinite(y[:, js[0]])
        xx, yy = x[rows], y[np.ix_(rows, js)]
        for ks in xg.values():
            ok = np.isfinite(xx[:, ks[0]])
            nn = int(ok.sum())
            n[np.ix_(ks, js)] = nn
            if nn < min_pairs:
                continue
            a = rankdata(xx[np.ix_(ok, ks)], axis=0, method="average")
            b = rankdata(yy[ok], axis=0, method="average")
            a -= a.mean(axis=0); b -= b.mean(axis=0)
            den = np.sqrt(np.sum(a*a, axis=0)[:, None] * np.sum(b*b, axis=0))
            r = np.divide(a.T @ b, den, out=np.zeros_like(den), where=den > 0)
            r = np.clip(r, -1., 1.)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.abs(r) * np.sqrt((nn-2)/np.maximum(1-r*r, 0))
            pp = 2*student_t.sf(z, nn-2)
            pp[den == 0] = 1
            rho[np.ix_(ks, js)] = r
            p[np.ix_(ks, js)] = pp
    q = false_discovery_control(p, axis=0).astype(np.float32)
    passed = (np.abs(rho) >= min_abs_rho) & (q <= max_fdr) & (n >= min_pairs)
    return dict(rho=rho, p=p, q=q, n=n, passed=passed)


def prepare(x, y, mask):
    """Fit X imputation/scaling on this target's observed training patients."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rows = np.isfinite(y)
    x, y = x[rows], y[rows]
    if len(y) < 2:
        raise ValueError("INSUFFICIENT_OBSERVED_PATIENTS")
    idx = np.flatnonzero(np.asarray(mask, bool) & np.isfinite(x).any(axis=0))
    if len(idx):
        raw = x[:, idx]
        med = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        a = np.where(np.isfinite(raw), raw, med)
        mean, sd = a.mean(axis=0), a.std(axis=0)
        keep = sd > 1e-12
        idx, med, mean, sd = idx[keep], med[keep], mean[keep], sd[keep]
        z = np.asfortranarray((a[:, keep]-mean)/sd)
    else:
        med = mean = sd = np.empty(0)
        z = np.empty((len(y), 0))
    mu, sy = float(y.mean()), float(y.std())
    return z, y, dict(feature_index=idx, median=med, mean=mean, scale=sd,
                      outcome_mean=mu, outcome_scale=sy)


def transform(x, state):
    a = np.asarray(x, float)[:, state["feature_index"]]
    return (np.where(np.isfinite(a), a, state["median"])-state["mean"])/state["scale"]


def path_predictions(x, y, mask, query, *, ratios=ALPHA_RATIOS, tol=1e-4, max_iter=20000):
    z, yy, st = prepare(x, y, mask)
    ratios = np.asarray(ratios, float)
    if len(ratios) == 0 or np.any(ratios <= 0) or np.any(ratios > 1) or np.any(np.diff(ratios) >= 0):
        raise ValueError("Alpha ratios must decrease from at most one")
    mu, sy = st["outcome_mean"], st["outcome_scale"]
    pred = np.full((len(query), len(ratios)), mu)
    if z.shape[1] == 0 or sy <= 1e-12:
        return pred, dict(n_screened=int(np.sum(mask)), n_variable=z.shape[1],
                          alpha_max=0., max_dual_gap=0., max_iterations=0)
    yn = (yy-mu)/sy
    amax = float(np.max(np.abs(z.T @ yn))/len(yn))
    if amax <= 1e-14:
        return pred, dict(n_screened=int(np.sum(mask)), n_variable=z.shape[1],
                          alpha_max=amax, max_dual_gap=0., max_iterations=0)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", ConvergenceWarning)
        alphas, coef, gaps, iters = solve_path(
            z, yn, alphas=amax*ratios, tol=tol, max_iter=max_iter,
            precompute=False, return_n_iter=True, selection="cyclic")
    if np.any(gaps > tol*1.01) or any(issubclass(w.category, ConvergenceWarning) for w in ws):
        with warnings.catch_warnings(record=True) as retry:
            warnings.simplefilter("always", ConvergenceWarning)
            alphas, coef, gaps, iters = solve_path(
                z, yn, alphas=amax*ratios, tol=tol, max_iter=max_iter*5,
                precompute=False, return_n_iter=True, selection="cyclic")
        if np.any(gaps > tol*1.01) or any(issubclass(w.category, ConvergenceWarning) for w in retry):
            raise RuntimeError("LASSO_PATH_NOT_CONVERGED")
    coef[:, ratios == 1.] = 0.
    pred = transform(query, st) @ coef * sy + mu
    return pred, dict(n_screened=int(np.sum(mask)), n_variable=z.shape[1],
                      alpha_max=amax, max_dual_gap=float(np.max(gaps)),
                      max_iterations=int(max(iters)))


def fit_fixed(x, y, mask, query, ratio, *, tol=1e-4, max_iter=20000):
    z, yy, st = prepare(x, y, mask)
    mu, sy = st["outcome_mean"], st["outcome_scale"]
    coef = np.zeros(z.shape[1]); alpha = gap = 0.; nit = 0
    if z.shape[1] and sy > 1e-12:
        yn = (yy-mu)/sy
        alpha = float(np.max(np.abs(z.T @ yn))/len(yn)*ratio)
        if alpha > 1e-14 and ratio < 1.:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", ConvergenceWarning)
                _, cc, gg, ii = solve_path(z, yn, alphas=np.array([alpha]), tol=tol,
                    max_iter=max_iter, precompute=False, return_n_iter=True, selection="cyclic")
            if gg[0] > tol*1.01 or any(issubclass(w.category, ConvergenceWarning) for w in ws):
                _, cc, gg, ii = solve_path(z, yn, alphas=np.array([alpha]), tol=tol,
                    max_iter=max_iter*5, precompute=False, return_n_iter=True, selection="cyclic")
            if gg[0] > tol*1.01:
                raise RuntimeError("LASSO_FIT_NOT_CONVERGED")
            coef = cc[:, 0]*sy
            gap, nit = float(gg[0]), int(ii[0])
    raw_coef = coef/st["scale"]
    pred = transform(query, st) @ coef + mu
    raw_intercept = mu-float(raw_coef @ st["mean"])
    state = dict(**st, coefficient=raw_coef, standardized_coefficient=coef,
                 intercept=raw_intercept, alpha=alpha, alpha_ratio=float(ratio),
                 dual_gap=gap, n_iter=nit)
    return pred, state


def predict_state(state, x):
    raw = np.asarray(x, float)[:, state["feature_index"]]
    raw = np.where(np.isfinite(raw), raw, state["median"])
    return raw @ state["coefficient"] + state["intercept"]


def fit_split(x_train, y_train, x_validation, y_validation, x_test, *,
              min_abs_rho=.20, max_fdr=.05, min_pairs=10, ratios=ALPHA_RATIOS):
    """Test Y is deliberately not an argument."""
    a = screen(x_train, y_train, min_abs_rho=min_abs_rho, max_fdr=max_fdr, min_pairs=min_pairs)
    pred, audit = path_predictions(x_train, y_train, a["passed"][:, 0], x_validation, ratios=ratios)
    ok = np.isfinite(y_validation)
    if ok.sum() < 3:
        raise ValueError("INSUFFICIENT_VALIDATION_PATIENTS")
    mse = np.mean((pred[ok]-np.asarray(y_validation)[ok, None])**2, axis=0)
    best = int(np.flatnonzero(mse <= np.min(mse)+1e-12*max(1.,np.min(mse)))[0])
    xx = np.concatenate([x_train, x_validation]); yy = np.r_[y_train, y_validation]
    b = screen(xx, yy, min_abs_rho=min_abs_rho, max_fdr=max_fdr, min_pairs=min_pairs)
    predictions, state = fit_fixed(xx, yy, b["passed"][:, 0], x_test, ratios[best])
    return predictions, state, dict(**audit, chosen_index=best, validation_mse=float(mse[best]),
                                     refit_screened=int(b["passed"].sum()))
