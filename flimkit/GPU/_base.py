import numpy as np


class GPUBackend:
    """Structural interface that every backend must satisfy.

    Backends don't need to inherit from this — it's here for documentation
    and static type-checking only.
    """

    def batch_fixed_tau(
        self,
        stack,                      # (ny, nx, n_bins) float
        A,                          # (n_bins, n_exp) — IRF-convolved basis
        taus_fixed,                 # (n_exp,) seconds
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback,
    ):
        """Fit all pixels at once using fixed lifetimes (n_exp ≥ 2)."""
        raise NotImplementedError

    def batch_grid_scan_1exp(
        self,
        stack,                      # (ny, nx, n_bins)
        basis_grid,                 # (N_GRID, n_bins)
        bb_grid,                    # (N_GRID,) — ||basis[j]||^2
        tau_grid,                   # (N_GRID,) seconds
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback,
    ):
        """Fit all pixels at once by scanning a log-spaced τ grid (n_exp == 1)."""
        raise NotImplementedError


class _BackendMixin:
    """NumPy-level helpers shared by TorchBackend and MLXBackend.

    Keeps output-map assembly and result scattering in one place so
    both backends stay in sync without duplicating logic.
    """
    @staticmethod
    def _init_maps(ny, nx, n_exp, intensity, taus_fixed_ns, free_tau):
        """Allocate output maps pre-filled with NaN, matching the CPU fitter layout."""
        maps = dict(
            intensity    = intensity,
            tau_mean_int = np.full((ny, nx), np.nan),
            tau_mean_amp = np.full((ny, nx), np.nan),
            chi2_r       = np.full((ny, nx), np.nan),
        )
        for i in range(n_exp):
            maps[f"alpha_{i+1}"] = np.full((ny, nx), np.nan)
            maps[f"frac_{i+1}"]  = np.full((ny, nx), np.nan)
            maps[f"tau_{i+1}"]   = (
                np.full((ny, nx), np.nan) if (n_exp == 1 or free_tau)
                else np.full((ny, nx), taus_fixed_ns[i])
            )
            maps[f"a{i+1}"] = maps[f"alpha_{i+1}"]   # alias
        return maps

    @staticmethod
    def _scatter_fixed_tau(
        maps,
        valid_idx,                  # (N_valid,) flat pixel indices
        amps,                       # (N_valid, n_exp) non-negative
        bg,                         # (N_valid,)
        decay_valid,                # (N_valid, n_bins) raw counts
        A,                          # (n_bins, n_exp)
        taus_ns,                    # (n_exp,)
        ny, nx,
    ):
        """Write per-pixel amplitude/lifetime/chi2 results into the maps dict."""
        n_exp = A.shape[1]
        n_bins = A.shape[0]

        amp_sum = amps.sum(axis=1)                      # (N_valid,)
        good    = amp_sum > 0
        if not good.any():
            return

        fracs    = np.where(good[:, None], amps / np.maximum(amp_sum[:, None], 1e-30), 0.0)
        taus_ns2 = taus_ns ** 2
        tau_amp  = (fracs * taus_ns[None, :]).sum(axis=1)   # (N_valid,)
        denom    = (amps * taus_ns[None, :]).sum(axis=1)
        tau_int  = np.where(denom > 0,
                            (amps * taus_ns2[None, :]).sum(axis=1) / np.maximum(denom, 1e-30),
                            np.nan)

        model    = decay_valid.copy()                    # allocate
        for j in range(n_exp):
            model = amps[:, j:j+1] * A[:, j][None, :]  # placeholder
        # Recompute properly: model = amps @ A.T + bg
        model    = amps @ A.T + bg[:, None]             # (N_valid, n_bins)
        resid    = decay_valid - model
        chi2     = (resid ** 2 / np.maximum(model, 1.0)).sum(axis=1)
        dof      = max(n_bins - n_exp, 1)

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))

        maps["tau_mean_amp"][yi_arr[good], xi_arr[good]] = tau_amp[good]
        maps["tau_mean_int"][yi_arr[good], xi_arr[good]] = tau_int[good]
        maps["chi2_r"][yi_arr[good], xi_arr[good]]       = chi2[good] / dof
        for i in range(n_exp):
            maps[f"alpha_{i+1}"][yi_arr[good], xi_arr[good]] = amps[good, i]
            maps[f"frac_{i+1}"][yi_arr[good], xi_arr[good]]  = fracs[good, i]

    @staticmethod
    def _scatter_1exp(
        maps,
        valid_idx,                  # (N_valid,) flat pixel indices
        tau_v,                      # (N_valid,) seconds
        amp_v,                      # (N_valid,)
        bg_v,                       # (N_valid,)
        decay_valid,                # (N_valid, n_bins)
        basis_best,                 # (N_valid, n_bins)
        ny, nx,
        n_bins,
    ):
        """Write single-exp grid-scan results into the maps dict."""
        good   = amp_v > 0
        tau_ns = tau_v * 1e9
        model  = amp_v[:, None] * basis_best + bg_v[:, None]
        resid  = decay_valid - model
        chi2   = (resid ** 2 / np.maximum(model, 1.0)).sum(axis=1) / max(n_bins - 2, 1)

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
        maps["tau_1"][yi_arr[good], xi_arr[good]]        = tau_ns[good]
        maps["tau_mean_amp"][yi_arr[good], xi_arr[good]] = tau_ns[good]
        maps["tau_mean_int"][yi_arr[good], xi_arr[good]] = tau_ns[good]
        maps["alpha_1"][yi_arr[good], xi_arr[good]]      = amp_v[good]
        maps["frac_1"][yi_arr[good], xi_arr[good]]       = 1.0
        maps["chi2_r"][yi_arr[good], xi_arr[good]]       = chi2[good]
