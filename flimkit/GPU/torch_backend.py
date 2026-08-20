import threading

import numpy as np
from flimkit.GPU._base import (_BackendMixin, fit_window, pixel_blocks,
                               gpu_block_bytes, CUDA_BLOCK_BYTES)
from flimkit.FLIM.fit_tools import (calibrated_chi2, distribution_dof,
                                    estimate_bg, coates_pileup_correction)

_MATMUL_PRECISION_LOCK = threading.Lock()

class TorchBackend(_BackendMixin):

    def __init__(self, device='cuda'):
        import torch
        self._torch = torch
        self.device = torch.device(device)

    def block_bytes(self):
        if self.device.type != 'cuda':
            return gpu_block_bytes()
        budget = gpu_block_bytes(CUDA_BLOCK_BYTES)
        try:
            free = int(self._torch.cuda.mem_get_info(self.device)[0])
        except Exception:
            return budget
        return max(1, min(budget, free // 2))

    def _matmul_full_precision(self, left, right):
        if self.device.type != 'cuda':
            return left @ right
        with _MATMUL_PRECISION_LOCK:
            previous = self._torch.get_float32_matmul_precision()
            self._torch.set_float32_matmul_precision('highest')
            try:
                return left @ right
            finally:
                self._torch.set_float32_matmul_precision(previous)

    def __repr__(self):
        return f"TorchBackend(device='{self.device}')"

    def batch_fixed_tau(
        self,
        stack,
        A,
        taus_fixed,
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback=None,
        tvb_profile=None,
        fit_tvb=False,
        fit_idx=None,
    ):
        torch = self._torch
        ny, nx, n_bins = stack.shape
        n_exp = A.shape[1]
        taus_ns = taus_fixed * 1e9
        raw = stack.reshape(ny * nx, n_bins)
        win = fit_window(fit_idx, n_bins)
        A = A if win is None else A[win]
        valid_idx = np.where(raw.sum(axis=1) >= min_photons)[0]
        maps = self._init_maps(
            ny, nx, n_exp,
            intensity = stack.sum(axis=2),
            taus_fixed_ns = taus_ns,
            free_tau = False,
        )
        if valid_idx.size == 0:
            return maps
        with_tvb = fit_tvb and tvb_profile is not None
        B_col = None
        if with_tvb:
            B_col = np.asarray(tvb_profile, dtype=np.float32)
            B_col = B_col if win is None else B_col[win]
            A_aug = np.column_stack(
                [A, B_col, np.ones(A.shape[0], dtype=np.float32)]).astype(np.float32)
            A_pinv = torch.linalg.pinv(torch.as_tensor(A_aug, device='cpu')).to(self.device)
        else:
            A_cpu = torch.as_tensor(A, dtype=torch.float32, device='cpu')
            A_pinv = torch.linalg.pinv(A_cpu).to(self.device)
        n_fit = A.shape[0]
        for first, last in pixel_blocks(valid_idx.size, 4 * (2 * n_bins + n_fit + n_exp),
                                        budget=self.block_bytes()):
            block = valid_idx[first:last]
            decay = raw[block].astype(np.float32)
            if with_tvb:
                data_in = (decay if win is None else decay[:, win]).copy()
                if correct_pileup and n_sync_px > 0:
                    for row in range(data_in.shape[0]):
                        data_in[row] = coates_pileup_correction(data_in[row], n_sync_px)
                data_t = torch.as_tensor(data_in, dtype=torch.float32, device=self.device)
                coeffs = torch.clamp(data_t @ A_pinv.T, min=0.0).cpu().numpy()
                amps = coeffs[:, :n_exp]
                tvb = coeffs[:, n_exp]
                bg = coeffs[:, n_exp + 1]
                decay_fit = data_in
            else:
                bg = self._estimate_bg_batch(decay, np.ones(decay.shape[0], dtype=bool))
                corrected = np.maximum(decay - bg[:, None], 0.0)
                corrected = corrected if win is None else corrected[:, win]
                if correct_pileup and n_sync_px > 0:
                    for row in range(corrected.shape[0]):
                        corrected[row] = coates_pileup_correction(corrected[row], n_sync_px)
                data_t = torch.as_tensor(corrected, dtype=torch.float32, device=self.device)
                amps = torch.clamp(data_t @ A_pinv.T, min=0.0).cpu().numpy()
                tvb = None
                decay_fit = decay if win is None else decay[:, win]
            self._scatter_fixed_tau(
                maps,
                valid_idx = block,
                amps = amps,
                bg = bg,
                decay_valid = decay_fit,
                A = A,
                taus_ns = taus_ns,
                ny = ny, nx = nx,
                tvb = tvb,
                tvb_profile = B_col if with_tvb else None,
            )
            if progress_callback is not None:
                progress_callback(last, valid_idx.size)
        return maps

    def batch_grid_scan_1exp(
        self,
        stack,
        basis_grid,
        bb_grid,
        tau_grid,
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback=None,
        tvb_profile=None,
        fit_tvb=False,
        fit_idx=None,
    ):
        torch = self._torch
        ny, nx, n_bins = stack.shape
        N_GRID = len(tau_grid)
        raw = stack.reshape(ny * nx, n_bins)
        win = fit_window(fit_idx, n_bins)
        if win is not None:
            if fit_tvb and tvb_profile is not None:
                raise ValueError('a fit window with time-varying background is not '
                                 'supported on the GPU for one-exponential fits')
            basis_grid = basis_grid[:, win]
            bb_grid = np.maximum((basis_grid ** 2).sum(axis=1), 1e-20)
            n_fit = len(win)
        else:
            n_fit = n_bins
        valid_idx = np.where(raw.sum(axis=1) >= min_photons)[0]
        maps = self._init_maps(
            ny, nx, n_exp=1,
            intensity=stack.sum(axis=2),
            taus_fixed_ns=np.array([tau_grid[N_GRID // 2] * 1e9]),
            free_tau=True,
        )
        if valid_idx.size == 0:
            return maps
        with_tvb = fit_tvb and tvb_profile is not None
        if with_tvb:
            U, U_pinv, basis_perp, bb_perp = self._tvb_grid_prep(
                basis_grid, tvb_profile, n_bins)
            basis_t = torch.as_tensor(basis_perp, dtype=torch.float32, device=self.device)
            bbp_t = torch.as_tensor(bb_perp, dtype=torch.float32, device=self.device)
        else:
            basis_t = torch.as_tensor(basis_grid, dtype=torch.float32, device=self.device)
            bb_t = torch.as_tensor(bb_grid, dtype=torch.float32, device=self.device)
        per_pixel = 4 * (2 * n_bins + n_fit + N_GRID)
        for first, last in pixel_blocks(valid_idx.size, per_pixel,
                                        budget=self.block_bytes()):
            block = valid_idx[first:last]
            decay = raw[block].astype(np.float32)
            if with_tvb:
                data_in = decay.copy()
                if correct_pileup and n_sync_px > 0:
                    for row in range(data_in.shape[0]):
                        data_in[row] = coates_pileup_correction(data_in[row], n_sync_px)
                d_perp = self._tvb_project_data(
                    data_in.astype(np.float64), U, U_pinv).astype(np.float32)
                dperp_t = torch.as_tensor(d_perp, dtype=torch.float32, device=self.device)
                bd = self._matmul_full_precision(dperp_t, basis_t.T)
                dsq = (dperp_t ** 2).sum(dim=1)
                costs = dsq[:, None] - torch.clamp(bd, min=0.0) ** 2 / bbp_t[None, :]
                best_g = costs.argmin(dim=1).cpu().numpy()
                bd_np = bd.cpu().numpy()
                amp_v = np.maximum(
                    bd_np[np.arange(block.size), best_g] / bb_perp[best_g], 0.0)
                basis_best = basis_grid[best_g]
                resid_after = data_in.astype(np.float64) - amp_v[:, None] * basis_best
                vz = resid_after @ U_pinv.T
                self._scatter_1exp(
                    maps, valid_idx=block, tau_v=tau_grid[best_g], amp_v=amp_v,
                    bg_v=vz[:, 1].astype(np.float32), decay_valid=data_in,
                    basis_best=basis_best, ny=ny, nx=nx, n_bins=n_bins,
                    tvb=np.maximum(vz[:, 0], 0.0).astype(np.float32),
                    tvb_profile=np.asarray(tvb_profile, dtype=np.float32),
                )
            else:
                bg = self._estimate_bg_batch(decay, np.ones(decay.shape[0], dtype=bool))
                corrected = np.maximum(decay - bg[:, None], 0.0)
                if correct_pileup and n_sync_px > 0:
                    for row in range(corrected.shape[0]):
                        corrected[row] = coates_pileup_correction(
                            corrected[row], n_sync_px)
                corrected = corrected if win is None else corrected[:, win]
                dc_t = torch.as_tensor(corrected, dtype=torch.float32, device=self.device)
                bd = dc_t @ basis_t.T
                dc_sq = (dc_t ** 2).sum(dim=1)
                costs = dc_sq[:, None] - torch.clamp(bd, min=0.0) ** 2 / bb_t[None, :]
                best_g = costs.argmin(dim=1).cpu().numpy()
                bd_np = bd.cpu().numpy()
                amp_v = np.maximum(
                    bd_np[np.arange(block.size), best_g] / bb_grid[best_g], 0.0)
                self._scatter_1exp(
                    maps,
                    valid_idx = block,
                    tau_v = tau_grid[best_g],
                    amp_v = amp_v,
                    bg_v = bg,
                    decay_valid = decay if win is None else decay[:, win],
                    basis_best = basis_grid[best_g],
                    ny = ny, nx = nx,
                    n_bins = n_fit,
                )
            if progress_callback is not None:
                progress_callback(last, valid_idx.size)
        return maps

    def batch_free_tau_fit(
        self,
        stack,
        irf_array,
        tcspc_res,
        taus_init,
        tau_min_s,
        tau_max_s,
        n_exp,
        min_photons,
        correct_pileup,
        n_sync_px,
        n_steps=50,
        lr=None,
        tvb_profile=None,
        fit_tvb=False,
        fit_idx=None,
        n_sync_model=None,
    ):
        torch = self._torch
        ny, nx, n_bins = stack.shape
        taus_ns_init = taus_init * 1e9
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons
        valid_idx = np.where(valid_mask)[0]
        maps = self._init_maps(
            ny, nx, n_exp,
            intensity=stack.sum(axis=2),
            taus_fixed_ns=taus_ns_init,
            free_tau=True,
        )
        if valid_idx.size == 0:
            return maps
        fit_flat = flat
        if correct_pileup and n_sync_px > 0:
            fit_flat = flat.copy()
            for idx in valid_idx:
                fit_flat[idx] = coates_pileup_correction(flat[idx], n_sync_px)
        bg_flat = self._estimate_bg_batch(fit_flat, valid_mask)
        raw_valid = fit_flat[valid_idx].astype(np.float32)
        bg_valid = bg_flat[valid_idx].astype(np.float32)
        weight_valid = flat[valid_idx].astype(np.float32)
        B = len(valid_idx)
        taus_out, amps_out, chi2r_out, chi2c_out, _, valid_b, tvb_out = self._scipy_parallel_free_tau_fit(
            raw_valid, bg_valid, irf_array, tcspc_res,
            taus_init, tau_min_s, tau_max_s, n_exp, n_bins,
            tvb_profile=tvb_profile, fit_tvb=fit_tvb, fit_idx=fit_idx,
            weight_valid=weight_valid, n_sync_model=n_sync_model,
        )
        self._scatter_free_tau(
            maps, valid_idx=valid_idx[valid_b],
            taus_s=taus_out[valid_b], amps=amps_out[valid_b],
            chi2_r=chi2r_out[valid_b], calibrated_values=chi2c_out[valid_b],
            ny=ny, nx=nx, n_exp=n_exp,
            tvb=tvb_out[valid_b] if fit_tvb else None,
        )
        return maps

    def batch_dist_scan_unimodal(
        self,
        stack,
        basis,
        bb_grid,
        param_pairs,
        irf_fixed,
        tcspc_res,
        n_bins,
        dist_type,
        min_photons,
        progress_callback=None,
        tvb_profile=None,
        fit_tvb=False,
        fit_idx=None,
    ):
        torch = self._torch
        ny, nx, _ = stack.shape
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        win = fit_window(fit_idx, n_bins)
        n_fit = n_bins if win is None else len(win)
        flat_fit = flat if win is None else flat[:, win]
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons
        valid_idx = np.where(valid_mask)[0]
        maps = dict(
            intensity = stack.sum(axis=2),
            tau_mean_amp = np.full((ny, nx), np.nan),
            tau_mean_int = np.full((ny, nx), np.nan),
            chi2_r = np.full((ny, nx), np.nan),
            calibrated_chi2_r = np.full((ny, nx), np.nan),
            tau_center_1 = np.full((ny, nx), np.nan),
            width_1 = np.full((ny, nx), np.nan),
            alpha_1 = np.full((ny, nx), np.nan),
            frac_1 = np.full((ny, nx), np.nan),
        )
        if valid_idx.size == 0:
            return maps
        if fit_tvb and tvb_profile is not None:
            maps['tvb_scale'] = np.full((ny, nx), np.nan)
            tvb_fit = np.asarray(tvb_profile) if win is None else np.asarray(tvb_profile)[win]
            U, U_pinv, basis_perp, bb_perp = self._tvb_grid_prep(basis, tvb_fit, n_fit)
            d_valid = flat_fit[valid_idx]
            d_perp = self._tvb_project_data(d_valid.astype(np.float64), U, U_pinv).astype(np.float32)
            basis_pt = torch.as_tensor(basis_perp, dtype=torch.float32, device=self.device)
            bbp_t = torch.as_tensor(bb_perp, dtype=torch.float32, device=self.device)
            dperp_t = torch.as_tensor(d_perp, dtype=torch.float32, device=self.device)
            bd_t = self._matmul_full_precision(dperp_t, basis_pt.T)
            dsq_t = (dperp_t ** 2).sum(dim=1)
            costs_t = dsq_t[:, None] - torch.clamp(bd_t, min=0.0) ** 2 / bbp_t[None, :]
            best_g = costs_t.argmin(dim=1).cpu().numpy()
            bd_np = bd_t.cpu().numpy()
            tau_v = param_pairs[best_g, 0]
            w_v = param_pairs[best_g, 1]
            amp_v = np.maximum(bd_np[np.arange(len(valid_idx)), best_g] / bb_perp[best_g], 0.0)
            good = amp_v > 0
            tau_amp_ns = tau_v * 1e9
            tau_int_ns = (tau_v + w_v ** 2 / np.maximum(tau_v, 1e-15)) * 1e9
            basis_best = basis[best_g].astype(np.float64)
            resid_after = d_valid.astype(np.float64) - amp_v[:, None] * basis_best
            vz = resid_after @ U_pinv.T
            tvb_v = np.maximum(vz[:, 0], 0.0)
            bg_z = vz[:, 1]
            B_arr = np.asarray(tvb_fit, dtype=np.float64)
            model_v = amp_v[:, None] * basis_best + tvb_v[:, None] * B_arr[None, :] + bg_z[:, None]
            resid_v = d_valid.astype(np.float64) - model_v
            chi2_v = (resid_v ** 2 / np.maximum(model_v, 1.0)).sum(axis=1) / distribution_dof(n_fit, 1, True)
            chi2_cal_v = calibrated_chi2(d_valid, model_v, axis=1)
            yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
            maps['tau_center_1'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
            maps['width_1'][yi_arr[good], xi_arr[good]] = w_v[good] * 1e9
            maps['alpha_1'][yi_arr[good], xi_arr[good]] = amp_v[good]
            maps['frac_1'][yi_arr[good], xi_arr[good]] = 1.0
            maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
            maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_int_ns[good]
            maps['chi2_r'][yi_arr[good], xi_arr[good]] = chi2_v[good]
            maps['calibrated_chi2_r'][yi_arr[good], xi_arr[good]] = chi2_cal_v[good]
            maps['tvb_scale'][yi_arr[good], xi_arr[good]] = tvb_v[good]
            return maps
        bg_flat = self._estimate_bg_batch(flat, valid_mask)
        dc_flat = np.maximum(flat_fit - bg_flat[:, None], 0.0)
        dc_valid = dc_flat[valid_idx]
        basis_t = torch.as_tensor(basis,    dtype=torch.float32, device=self.device)
        bb_t = torch.as_tensor(bb_grid,  dtype=torch.float32, device=self.device)
        dc_t = torch.as_tensor(dc_valid, dtype=torch.float32, device=self.device)
        bd_t = dc_t @ basis_t.T
        dc_sq_t = (dc_t ** 2).sum(dim=1)
        costs_t = dc_sq_t[:, None] - torch.clamp(bd_t, min=0.0) ** 2 / bb_t[None, :]
        best_g = costs_t.argmin(dim=1).cpu().numpy()
        bd_np = bd_t.cpu().numpy()
        tau_v = param_pairs[best_g, 0]
        w_v = param_pairs[best_g, 1]
        amp_v = np.maximum(
            bd_np[np.arange(len(valid_idx)), best_g] / bb_grid[best_g].astype(np.float64), 0.0)
        good = amp_v > 0
        tau_amp_ns = tau_v * 1e9
        tau_int_ns = (tau_v + w_v ** 2 / np.maximum(tau_v, 1e-15)) * 1e9
        basis_best = basis[best_g].astype(np.float64)
        model_v = amp_v[:, None] * basis_best + bg_flat[valid_idx, None]
        resid_v = flat_fit[valid_idx].astype(np.float64) - model_v
        chi2_v = (resid_v ** 2 / np.maximum(model_v, 1.0)).sum(axis=1) / distribution_dof(n_fit, 1, False)
        chi2_cal_v = calibrated_chi2(flat_fit[valid_idx], model_v, axis=1)
        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
        maps['tau_center_1'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
        maps['width_1'][yi_arr[good], xi_arr[good]] = w_v[good] * 1e9
        maps['alpha_1'][yi_arr[good], xi_arr[good]] = amp_v[good]
        maps['frac_1'][yi_arr[good], xi_arr[good]] = 1.0
        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_int_ns[good]
        maps['chi2_r'][yi_arr[good], xi_arr[good]] = chi2_v[good]
        maps['calibrated_chi2_r'][yi_arr[good], xi_arr[good]] = chi2_cal_v[good]
        return maps
