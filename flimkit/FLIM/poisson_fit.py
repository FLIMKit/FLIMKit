import os
import numpy as np
import numba as nb

def _avoid_a_second_openmp_runtime():
    if 'NUMBA_THREADING_LAYER' not in os.environ:
        nb.config.THREADING_LAYER = 'workqueue'

_avoid_a_second_openmp_runtime()

@nb.njit(parallel=True, cache=True)
def _em_sparse(y, M, colsum, iters, tol, out):
    n_px, n_bins = y.shape
    n_col = M.shape[1]
    for p in nb.prange(n_px):
        idx = np.empty(n_bins, np.int64)
        m = 0
        tot = 0.0
        for b in range(n_bins):
            if y[p, b] > 0:
                idx[m] = b
                m += 1
                tot += y[p, b]
        if m == 0:
            continue
        a = np.empty(n_col)
        num = np.empty(n_col)
        for k in range(n_col):
            a[k] = tot / n_col / colsum[k]
        for it in range(iters):
            for k in range(n_col):
                num[k] = 0.0
            for j in range(m):
                b = idx[j]
                mu = 0.0
                for k in range(n_col):
                    mu += a[k] * M[b, k]
                if mu < 1e-12:
                    mu = 1e-12
                r = y[p, b] / mu
                for k in range(n_col):
                    num[k] += r * M[b, k]
            dmax = 0.0
            for k in range(n_col):
                new = a[k] * num[k] / colsum[k]
                d = abs(new - a[k]) / (a[k] + 1e-12)
                if d > dmax:
                    dmax = d
                a[k] = new
            if dmax < tol:
                break
        for k in range(n_col):
            out[p, k] = a[k]

def poisson_amplitudes(counts, A, offset=True, iters=1500, tol=1e-7):
    y = np.ascontiguousarray(counts, dtype=np.float64)
    n_fit = A.shape[0]
    M = np.column_stack([A, np.ones(n_fit) / n_fit]) if offset else np.asarray(A, dtype=np.float64)
    M = np.ascontiguousarray(M, dtype=np.float64)
    out = np.zeros((y.shape[0], M.shape[1]))
    _em_sparse(y, M, M.sum(axis=0), int(iters), float(tol), out)
    if offset:
        return out[:, :-1], out[:, -1] / n_fit
    return out, np.zeros(y.shape[0])

def fitFixedTauPoisson(stack, A, taus_fixed, min_photons, progress_callback=None,
                       fit_idx=None, offset=True, iters=1500, tol=1e-7):
    from ..GPU._base import _BackendMixin, fit_window, pixel_blocks
    ny, nx, n_bins = stack.shape
    n_exp = A.shape[1]
    taus_ns = np.asarray(taus_fixed, dtype=float) * 1e9
    raw = stack.reshape(ny * nx, n_bins)
    win = fit_window(fit_idx, n_bins)
    A_fit = np.asarray(A, dtype=np.float64)
    A_fit = A_fit if win is None else A_fit[win]
    valid_idx = np.where(raw.sum(axis=1) >= min_photons)[0]
    maps = _BackendMixin._init_maps(ny, nx, n_exp, stack.sum(axis=2), taus_ns, False)
    if valid_idx.size == 0:
        return maps
    n_fit = A_fit.shape[0]
    for first, last in pixel_blocks(valid_idx.size, 8 * (n_bins + n_fit + 2 * n_exp)):
        block = valid_idx[first:last]
        decay = raw[block].astype(np.float64)
        decay = decay if win is None else decay[:, win]
        amps, bg = poisson_amplitudes(decay, A_fit, offset, iters, tol)
        _BackendMixin._scatter_fixed_tau(maps, valid_idx=block, amps=amps, bg=bg,
                                         decay_valid=decay, A=A_fit, taus_ns=taus_ns,
                                         ny=ny, nx=nx)
        if progress_callback is not None:
            progress_callback(last, valid_idx.size)
    return maps
