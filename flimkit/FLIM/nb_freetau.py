# Per-pixel free-tau reconvolution fit compiled with numba, no C compiler needed.
# The circular convolution of r^k with the IRF obeys y[i] = r y[i-1] + (1 - r^n) irf[i],
# so a model evaluation is O(n_bins) instead of an FFT.
import math
import numpy as np
import numba as nb

@nb.njit(cache=True)
def _exp_conv(irf, r, y, z):
    n = irf.size
    rn = r ** n
    rn1 = r ** (n - 1)
    y0 = irf[0]
    z0 = 0.0
    rk = 1.0
    for k in range(1, n):
        rk1 = rk
        rk *= r
        y0 += rk * irf[n - k]
        z0 += k * rk1 * irf[n - k]
    y[0] = y0
    z[0] = z0
    for i in range(1, n):
        y[i] = r * y[i - 1] + (1 - rn) * irf[i]
        z[i] = y[i - 1] + r * z[i - 1] - n * rn1 * irf[i]

@nb.njit(cache=True)
def _cholesky_solve(A, b):
    n = b.size
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= A[j, k] * A[j, k]
        if (s > 0.0) == False:
            return False
        dd = math.sqrt(s)
        A[j, j] = dd
        for i in range(j + 1, n):
            t = A[i, j]
            for k in range(j):
                t -= A[i, k] * A[j, k]
            A[i, j] = t / dd
    for i in range(n):
        t = b[i]
        for k in range(i):
            t -= A[i, k] * b[k]
        b[i] = t / A[i, i]
    for i in range(n - 1, -1, -1):
        t = b[i]
        for k in range(i + 1, n):
            t -= A[k, i] * b[k]
        b[i] = t / A[i, i]
    return True

@nb.njit(cache=True)
def _evaluate(p, ne, d, w, bg, irf, isum, idx, dt, Y, Z, res, J, want_J):
    drdt = np.empty(ne)
    for j in range(ne):
        tau = max(p[j], 1e-6)
        r = math.exp(-dt / tau)
        drdt[j] = r * dt / (tau * tau)
        _exp_conv(irf, r, Y[j], Z[j])
    cost = 0.0
    for q in range(idx.size):
        i = idx[q]
        m = bg * isum
        for j in range(ne):
            m += p[ne + j] * Y[j, i]
        ri = (m - d[i]) / w[i]
        cost += ri * ri
        if want_J == True:
            res[q] = ri
            for j in range(ne):
                J[q, j] = p[ne + j] * Z[j, i] * drdt[j] / w[i]
                J[q, ne + j] = Y[j, i] / w[i]
    return cost

@nb.njit(cache=True, parallel=True)
def fitFreeTau(data, w_data, bg, irf, idx, dt, p0, lo, hi, max_iter, tol):
    """Levenberg-Marquardt per row of data (B x n_bins) on the bins in idx, with
    weights sqrt(max(w_data, 1)). Params are taus (units of dt) then amplitudes,
    kept in [lo, hi]; one on a bound is held there while the step points out. Returns (params, cost, iterations), iterations -1 when
    the cost is not finite.
    """
    B, n = data.shape
    npar = p0.size
    ne = npar // 2
    isum = irf.sum()
    out_p = np.empty((B, npar))
    out_c = np.empty(B)
    out_i = np.empty(B, dtype=np.int32)
    for b in nb.prange(B):
        d = data[b]
        w = np.sqrt(np.maximum(w_data[b], 1.0))
        Y = np.empty((ne, n))
        Z = np.empty((ne, n))
        res = np.empty(idx.size)
        J = np.empty((idx.size, npar))
        p = np.minimum(np.maximum(p0.copy(), lo), hi)
        lam = 1e-3
        cost = _evaluate(p, ne, d, w, bg[b], irf, isum, idx, dt, Y, Z, res, J, True)
        n_it = max_iter
        for it in range(max_iter):
            A = J.T @ J
            g = -(J.T @ res)
            accepted = 0
            while lam < 1e12:
                M = A.copy()
                for k in range(npar):
                    M[k, k] += lam * max(A[k, k], 1e-30)
                step = g.copy()
                # a parameter on a bound that the step would push past is held there
                for k in range(npar):
                    if (p[k] <= lo[k] and g[k] < 0.0) or (p[k] >= hi[k] and g[k] > 0.0):
                        for l in range(npar):
                            M[k, l] = 0.0
                            M[l, k] = 0.0
                        M[k, k] = 1.0
                        step[k] = 0.0
                if _cholesky_solve(M, step) == True:
                    pt = np.minimum(np.maximum(p + step, lo), hi)
                    ct = _evaluate(pt, ne, d, w, bg[b], irf, isum, idx, dt, Y, Z, res, J, False)
                    if ct < cost:
                        rel = (cost - ct) / max(cost, 1e-300)
                        dp = ((pt - p) ** 2).sum()
                        pn = (p ** 2).sum()
                        p = pt
                        cost = _evaluate(p, ne, d, w, bg[b], irf, isum, idx, dt, Y, Z, res, J, True)
                        lam = max(lam * 0.1, 1e-12)
                        accepted = 2 if (rel < tol or dp <= tol * tol * (pn + tol)) else 1
                        break
                lam *= 10
            if accepted != 1:
                n_it = it
                break
        out_p[b] = p
        out_c[b] = cost
        out_i[b] = n_it if math.isfinite(cost) else -1
    return out_p, out_c, out_i

@nb.njit(cache=True, parallel=True)
def estimateBgRows(data, pre_gap=5):
    # fit_tools.estimate_bg for every row
    B = data.shape[0]
    out = np.empty(B)
    for b in nb.prange(B):
        row = data[b]
        end = max(0, int(np.argmax(row)) - pre_gap)
        if end >= 5:
            v = np.median(row[:end])
        else:
            v = np.median(row[-30:])
        out[b] = max(v, 0.0)
    return out

@nb.njit(cache=True, parallel=True)
def chi2Terms(params, data, bg, irf, idx, dt):
    # fit_tools.chi2_terms of each row against its fitted model, on the bins in idx
    B, n = data.shape
    ne = params.shape[1] // 2
    isum = irf.sum()
    numerator = np.empty(B)
    expected = np.empty(B)
    valid = np.empty(B, dtype=np.bool_)
    for b in nb.prange(B):
        Y = np.empty((ne, n))
        Z = np.empty((ne, n))
        for j in range(ne):
            tau = max(params[b, j], 1e-6)
            _exp_conv(irf, math.exp(-dt / tau), Y[j], Z[j])
        num = 0.0
        ex = 0.0
        ok = True
        for q in range(idx.size):
            i = idx[q]
            m = bg[b] * isum
            for j in range(ne):
                m += params[b, ne + j] * Y[j, i]
            d = data[b, i]
            if (math.isfinite(d) and d >= 0.0 and math.isfinite(m) and m >= 0.0) == False:
                ok = False
            num += (d - m) ** 2 / max(m, 1.0)
            ex += min(m, 1.0)
        numerator[b] = num
        expected[b] = ex
        valid[b] = ok
    return numerator, expected, valid
