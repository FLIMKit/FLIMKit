# nb_freetau.fitFreeTau on an NVIDIA GPU through numba.cuda, in double precision.
# Each launch gives every unfinished pixel a bounded number of LM steps and saves
# its state, so no launch outlasts the Windows display watchdog (about 2 s) and
# finished pixels drop out between launches. The step sequence is the CPU one.
import math
import time
import numpy as np
from numba import cuda, float64

MAX_EXP = 3
MAX_P = 6
MAX_PP = 36

def available():
    try:
        return cuda.is_available() == True
    except Exception:
        return False

@cuda.jit(device=True, inline=True)
def _evaluate(irf, mask, n, ne, irf_sum, dt, p, data, w_data, b, bg, A, g):
    np_ = 2 * ne
    r = cuda.local.array(MAX_EXP, float64)
    drdt = cuda.local.array(MAX_EXP, float64)
    y = cuda.local.array(MAX_EXP, float64)
    z = cuda.local.array(MAX_EXP, float64)
    rn = cuda.local.array(MAX_EXP, float64)
    rn1 = cuda.local.array(MAX_EXP, float64)
    jq = cuda.local.array(MAX_P, float64)
    for j in range(ne):
        tau = p[j] if p[j] > 1e-6 else 1e-6
        r[j] = math.exp(-dt / tau)
        drdt[j] = r[j] * dt / (tau * tau)
        rn[j] = math.pow(r[j], float(n))
        rn1[j] = math.pow(r[j], float(n - 1))
        y0 = irf[0]
        z0 = 0.0
        rk = 1.0
        for k in range(1, n):
            rk1 = rk
            rk *= r[j]
            y0 += rk * irf[n - k]
            z0 += k * rk1 * irf[n - k]
        y[j] = y0
        z[j] = z0
    for k in range(np_):
        g[k] = 0.0
        for l in range(np_):
            A[k * np_ + l] = 0.0
    base = bg * irf_sum
    cost = 0.0
    for i in range(n):
        if i > 0:
            ii = irf[i]
            for j in range(ne):
                yp = y[j]
                y[j] = r[j] * yp + (1 - rn[j]) * ii
                z[j] = yp + r[j] * z[j] - n * rn1[j] * ii
        if mask[i] == 0:
            continue
        wd = w_data[b, i]
        wi = math.sqrt(wd if wd > 1.0 else 1.0)
        m = base
        for j in range(ne):
            m += p[ne + j] * y[j]
        ri = (m - data[b, i]) / wi
        cost += ri * ri
        for j in range(ne):
            jq[j] = p[ne + j] * z[j] * drdt[j] / wi
            jq[ne + j] = y[j] / wi
        for k in range(np_):
            g[k] -= jq[k] * ri
            for l in range(k + 1):
                A[k * np_ + l] += jq[k] * jq[l]
    for k in range(np_):
        for l in range(k):
            A[l * np_ + k] = A[k * np_ + l]
    return cost

@cuda.jit(device=True, inline=True)
def _cholesky_solve(A, bv, n):
    for j in range(n):
        s = A[j * n + j]
        for k in range(j):
            s -= A[j * n + k] * A[j * n + k]
        if (s > 0.0) == False:
            return False
        dd = math.sqrt(s)
        A[j * n + j] = dd
        for i in range(j + 1, n):
            t = A[i * n + j]
            for k in range(j):
                t -= A[i * n + k] * A[j * n + k]
            A[i * n + j] = t / dd
    for i in range(n):
        t = bv[i]
        for k in range(i):
            t -= A[i * n + k] * bv[k]
        bv[i] = t / A[i * n + i]
    for i in range(n - 1, -1, -1):
        t = bv[i]
        for k in range(i + 1, n):
            t -= A[k * n + i] * bv[k]
        bv[i] = t / A[i * n + i]
    return True

@cuda.jit
def _fit_steps(n, ne, data, w_data, bg, irf, mask, irf_sum, dt, bd, max_iter, tol, max_steps,
               active, n_active, P, Ast, gst, cst, lamst, itst, flags, out_cost, out_iter):
    t = cuda.grid(1)
    if t >= n_active:
        return
    b = active[t]
    np_ = 2 * ne
    p = cuda.local.array(MAX_P, float64)
    pt = cuda.local.array(MAX_P, float64)
    g = cuda.local.array(MAX_P, float64)
    g2 = cuda.local.array(MAX_P, float64)
    step = cuda.local.array(MAX_P, float64)
    lo = cuda.local.array(MAX_P, float64)
    hi = cuda.local.array(MAX_P, float64)
    A = cuda.local.array(MAX_PP, float64)
    A2 = cuda.local.array(MAX_PP, float64)
    M = cuda.local.array(MAX_PP, float64)
    for k in range(np_):
        lo[k] = bd[MAX_P + k]
        hi[k] = bd[2 * MAX_P + k]
    fresh = flags[b] == 0
    cost = 0.0
    lam = 0.0
    it = 0
    if fresh == True:
        for k in range(np_):
            v = bd[k]
            p[k] = lo[k] if v < lo[k] else (hi[k] if v > hi[k] else v)
            pt[k] = p[k]
    else:
        for k in range(np_):
            p[k] = P[b, k]
            g[k] = gst[b, k]
        for k in range(np_ * np_):
            A[k] = Ast[b, k]
        cost = cst[b]
        lam = lamst[b]
        it = itst[b]
    done = False
    steps = 0
    while steps < max_steps:
        steps += 1
        if fresh == False:
            if lam >= 1e12:
                done = True
                break
            for k in range(np_ * np_):
                M[k] = A[k]
            for k in range(np_):
                dk = A[k * np_ + k]
                M[k * np_ + k] += lam * (dk if dk > 1e-30 else 1e-30)
                step[k] = g[k]
            for k in range(np_):
                if (p[k] <= lo[k] and g[k] < 0.0) or (p[k] >= hi[k] and g[k] > 0.0):
                    for l in range(np_):
                        M[k * np_ + l] = 0.0
                        M[l * np_ + k] = 0.0
                    M[k * np_ + k] = 1.0
                    step[k] = 0.0
            if _cholesky_solve(M, step, np_) == False:
                lam *= 10.0
                continue
            for k in range(np_):
                v = p[k] + step[k]
                pt[k] = lo[k] if v < lo[k] else (hi[k] if v > hi[k] else v)
        ct = _evaluate(irf, mask, n, ne, irf_sum, dt, pt, data, w_data, b, bg[b], A2, g2)
        if fresh == True:
            cost = ct
            for k in range(np_ * np_):
                A[k] = A2[k]
            for k in range(np_):
                g[k] = g2[k]
            lam = 1e-3
            it = 0
            fresh = False
        elif ct < cost:
            rel = (cost - ct) / (cost if cost > 1e-300 else 1e-300)
            dp = 0.0
            pn = 0.0
            for k in range(np_):
                dp += (pt[k] - p[k]) * (pt[k] - p[k])
                pn += p[k] * p[k]
                p[k] = pt[k]
            cost = ct
            for k in range(np_ * np_):
                A[k] = A2[k]
            for k in range(np_):
                g[k] = g2[k]
            lam = lam * 0.1 if lam * 0.1 > 1e-12 else 1e-12
            if rel < tol or dp <= tol * tol * (pn + tol):
                done = True
                break
            it += 1
            if it >= max_iter:
                done = True
                break
        else:
            lam *= 10.0
    for k in range(np_):
        P[b, k] = p[k]
    if done == True:
        flags[b] = 2
        out_cost[b] = cost
        out_iter[b] = it if math.isfinite(cost) else -1
    else:
        flags[b] = 1
        for k in range(np_):
            gst[b, k] = g[k]
        for k in range(np_ * np_):
            Ast[b, k] = A[k]
        cst[b] = cost
        lamst[b] = lam
        itst[b] = it

def fitFreeTau(data, w_data, bg, irf, idx, dt, p0, lo, hi, max_iter, tol, launch_s=0.3, block=128):
    """Same contract as nb_freetau.fitFreeTau. Launches are sized so each takes
    about launch_s seconds: steps per pixel and pixels per launch both shrink
    when a launch runs long.
    """
    B, n = data.shape
    ne = len(p0) // 2
    mask = np.zeros(n, dtype=np.int32)
    mask[np.asarray(idx)] = 1
    bd = np.zeros(3 * MAX_P)
    bd[:2 * ne] = p0
    bd[MAX_P:MAX_P + 2 * ne] = lo
    bd[2 * MAX_P:2 * MAX_P + 2 * ne] = hi
    d_data = cuda.to_device(np.ascontiguousarray(data, dtype=np.float64))
    d_w = d_data if w_data is data else cuda.to_device(np.ascontiguousarray(w_data, dtype=np.float64))
    d_bg = cuda.to_device(np.ascontiguousarray(bg, dtype=np.float64))
    d_irf = cuda.to_device(np.ascontiguousarray(irf, dtype=np.float64))
    d_mask = cuda.to_device(mask)
    d_bd = cuda.to_device(bd)
    P = cuda.device_array((B, MAX_P))
    Ast = cuda.device_array((B, MAX_PP))
    gst = cuda.device_array((B, MAX_P))
    cst = cuda.device_array(B)
    lamst = cuda.device_array(B)
    itst = cuda.device_array(B, dtype=np.int32)
    flags = cuda.to_device(np.zeros(B, dtype=np.int8))
    d_c = cuda.device_array(B)
    d_it = cuda.device_array(B, dtype=np.int32)
    irf_sum = float(np.sum(irf))
    args = (n, ne, d_data, d_w, d_bg, d_irf, d_mask, irf_sum, float(dt), d_bd, int(max_iter), float(tol))
    active = np.arange(B, dtype=np.int32)
    # an empty launch compiles the kernel before any launch is timed
    _fit_steps[1, block](*args, 4, cuda.to_device(active[:1]), 0, P, Ast, gst, cst, lamst, itst,
                         flags, d_c, d_it)
    cuda.synchronize()
    max_steps = 4
    max_px = B
    while active.size > 0:
        part = active[:max_px]
        t0 = time.perf_counter()
        _fit_steps[(part.size + block - 1) // block, block](
            *args, max_steps, cuda.to_device(part), part.size, P, Ast, gst, cst, lamst, itst,
            flags, d_c, d_it)
        cuda.synchronize()
        s = time.perf_counter() - t0
        if s < launch_s / 2 and part.size == active.size:
            max_steps = min(max_steps * 2, 256)
        elif s > launch_s:
            if max_steps > 1:
                max_steps = max(1, max_steps // 2)
            else:
                max_px = max(block, part.size // 2)
        if part.size < active.size:
            done = flags.copy_to_host()[part] == 2
            active = np.concatenate([active[part.size:], part[done == False]])
        else:
            active = np.flatnonzero(flags.copy_to_host() != 2).astype(np.int32)
    return P.copy_to_host()[:, :2 * ne], d_c.copy_to_host(), d_it.copy_to_host()
