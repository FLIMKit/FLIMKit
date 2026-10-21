import numpy as np

MAX_LAM = 1e12
MIN_LAM = 1e-12

def _conv_pair(torch, irf_f, taus, dt, n_bins):
    k = torch.arange(n_bins, dtype=taus.dtype, device=taus.device)
    r = torch.exp(-dt / torch.clamp(taus, min=1e-6))
    drdt = r * dt / (torch.clamp(taus, min=1e-6) ** 2)
    log_r = torch.log(torch.clamp(r, min=1e-300))
    pow_r = torch.exp(log_r[..., None] * k)
    dpow = k * torch.exp(log_r[..., None] * torch.clamp(k - 1.0, min=0.0))
    dpow[..., 0] = 0.0
    spec = torch.fft.rfft(torch.stack([pow_r, dpow], dim=0), dim=-1)
    both = torch.fft.irfft(spec * irf_f, n=n_bins, dim=-1)
    return both[0], both[1], drdt

def _evaluate(torch, p, ne, data, w, bg, irf_f, irf_sum, idx, dt, n_bins, want_jac):
    taus = p[:, :ne]
    amps = p[:, ne:2 * ne]
    y, z, drdt = _conv_pair(torch, irf_f, taus, dt, n_bins)
    y_fit = y[..., idx]
    model = bg[:, None] * irf_sum + (amps[..., None] * y_fit).sum(dim=1)
    res = (model - data) / w
    cost = (res * res).sum(dim=1)
    if want_jac == False:
        return cost, None, None
    z_fit = z[..., idx]
    jac = torch.cat([(amps * drdt)[..., None] * z_fit, y_fit], dim=1)
    jac = (jac / w[:, None, :]).transpose(1, 2)
    gram = jac.transpose(1, 2) @ jac
    grad = -(jac.transpose(1, 2) @ res[:, :, None])[:, :, 0]
    return cost, gram, grad

def _project(torch, gram, grad, p, lo, hi, lam):
    npar = p.shape[1]
    eye = torch.eye(npar, dtype=gram.dtype, device=gram.device)
    diag = torch.diagonal(gram, dim1=1, dim2=2)
    damped = gram + (lam[:, None] * torch.clamp(diag, min=1e-30))[:, :, None] * eye
    pinned = ((p <= lo) & (grad < 0.0)) | ((p >= hi) & (grad > 0.0))
    keep = (pinned == False).to(gram.dtype)
    damped = damped * keep[:, :, None] * keep[:, None, :]
    damped = damped + pinned.to(gram.dtype)[:, :, None] * eye
    return damped, torch.where(pinned, torch.zeros_like(grad), grad)

def fit_free_tau(torch, device, data, w_data, bg, irf, idx, dt, p0, lo, hi,
                 max_iter=200, tol=1e-8, dtype=None):
    dtype = torch.float64 if dtype is None else dtype
    n_bins = data.shape[1]
    d = torch.as_tensor(np.ascontiguousarray(data), dtype=dtype, device=device)
    w = torch.sqrt(torch.clamp(
        torch.as_tensor(np.ascontiguousarray(w_data), dtype=dtype, device=device), min=1.0))
    bg_t = torch.as_tensor(np.ascontiguousarray(bg), dtype=dtype, device=device)
    irf_t = torch.as_tensor(np.ascontiguousarray(irf), dtype=dtype, device=device)
    idx_t = torch.as_tensor(np.ascontiguousarray(idx), dtype=torch.long, device=device)
    lo_t = torch.as_tensor(np.ascontiguousarray(lo), dtype=dtype, device=device)
    hi_t = torch.as_tensor(np.ascontiguousarray(hi), dtype=dtype, device=device)
    p = torch.clamp(
        torch.as_tensor(np.ascontiguousarray(p0), dtype=dtype, device=device)
        .expand(d.shape[0], -1).clone(), lo_t, hi_t)
    irf_f = torch.fft.rfft(irf_t)
    irf_sum = irf_t.sum()
    ne = p.shape[1] // 2
    d_fit = d[:, idx_t]
    w_fit = w[:, idx_t]
    lo_b = lo_t.expand_as(p)
    hi_b = hi_t.expand_as(p)
    cost, gram, grad = _evaluate(torch, p, ne, d_fit, w_fit, bg_t, irf_f, irf_sum,
                                 idx_t, dt, n_bins, True)
    lam = torch.full_like(cost, 1e-3)
    iters = torch.zeros(d.shape[0], dtype=torch.int32, device=device)
    live = torch.ones(d.shape[0], dtype=torch.bool, device=device)
    for _ in range(max_iter * 24):
        rows = torch.nonzero(live, as_tuple=False)[:, 0]
        if rows.numel() == 0:
            break
        damped, step = _project(torch, gram[rows], grad[rows], p[rows],
                                lo_b[rows], hi_b[rows], lam[rows])
        chol, info = torch.linalg.cholesky_ex(damped)
        solved = info == 0
        step = torch.cholesky_solve(step[:, :, None], chol)[:, :, 0]
        trial = torch.clamp(p[rows] + step, lo_b[rows], hi_b[rows])
        trial = torch.where(solved[:, None], trial, p[rows])
        ct, _, _ = _evaluate(torch, trial, ne, d_fit[rows], w_fit[rows], bg_t[rows],
                             irf_f, irf_sum, idx_t, dt, n_bins, False)
        better = solved & (ct < cost[rows])
        rel = (cost[rows] - ct) / torch.clamp(cost[rows], min=1e-300)
        move = ((trial - p[rows]) ** 2).sum(dim=1)
        norm = (p[rows] ** 2).sum(dim=1)
        small = (rel < tol) | (move <= tol * tol * (norm + tol))
        taken = rows[better]
        if taken.numel() > 0:
            p[taken] = trial[better]
            cost[taken] = ct[better]
            iters[taken] = iters[taken] + 1
            lam[taken] = torch.clamp(lam[taken] * 0.1, min=MIN_LAM)
        missed = rows[better == False]
        if missed.numel() > 0:
            lam[missed] = lam[missed] * 10.0
        stop = torch.zeros_like(better)
        stop[better] = small[better]
        live[rows[stop]] = False
        live[missed[lam[missed] >= MAX_LAM]] = False
        live[rows[iters[rows] >= max_iter]] = False
        again = torch.nonzero(live, as_tuple=False)[:, 0]
        if again.numel() > 0:
            _, g2, gr2 = _evaluate(torch, p[again], ne, d_fit[again], w_fit[again],
                                   bg_t[again], irf_f, irf_sum, idx_t, dt, n_bins, True)
            gram[again] = g2
            grad[again] = gr2
    model_bins = _model_at(torch, p, ne, bg_t, irf_f, irf_sum, idx_t, dt, n_bins)
    num = ((d_fit - model_bins) ** 2 / torch.clamp(model_bins, min=1.0)).sum(dim=1)
    expected = torch.clamp(model_bins, max=1.0).sum(dim=1)
    ok = (torch.isfinite(d_fit) & (d_fit >= 0)
          & torch.isfinite(model_bins) & (model_bins >= 0)).all(dim=1)
    ok = ok & torch.isfinite(cost)
    return (p.cpu().numpy(), cost.cpu().numpy(), iters.cpu().numpy(),
            num.cpu().numpy(), expected.cpu().numpy(), ok.cpu().numpy())

def _model_at(torch, p, ne, bg, irf_f, irf_sum, idx, dt, n_bins):
    y, _, _ = _conv_pair(torch, irf_f, p[:, :ne], dt, n_bins)
    return bg[:, None] * irf_sum + (p[:, ne:2 * ne][..., None] * y[..., idx]).sum(dim=1)
