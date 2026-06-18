import numpy as np
from ..PTU.reader import PTUFile

def _resample_decay(decay, src_res, dst_res, n_bins):
    decay = np.asarray(decay, dtype=float)
    src_t = np.arange(decay.size, dtype=float) * src_res
    dst_t = np.arange(n_bins, dtype=float) * dst_res
    return np.interp(dst_t, src_t, decay)

def _fit_to_grid(decay, n_bins):
    decay = np.asarray(decay, dtype=float)
    if decay.size < n_bins:
        return np.concatenate([decay, np.zeros(n_bins - decay.size)])
    return decay[:n_bins]

def _normalize_profile(decay, normalize):
    decay = np.asarray(decay, dtype=float)
    total = float(decay.sum())
    if total <= 0:
        raise ValueError('Background profile has no photons.')
    if normalize:
        return decay / total
    return decay

def tvb_from_reference_ptu(path, ptu_ref, channel=None, normalize=True):
    ref   = PTUFile(path, verbose=False)
    decay = ref.summed_decay(channel=channel)
    total = decay.sum()
    if total == 0 and channel is not None:
        decay = ref.summed_decay(channel=None)
        total = decay.sum()
        if total > 0:
            print(f"  Background PTU has no photons on channel {channel}; "
                  f"using auto-detected channel {ref.photon_channel} instead")
    if total == 0:
        raise ValueError(f"Background PTU {path!r} has no photons.")
    if abs(ref.tcspc_res - ptu_ref.tcspc_res) > 1e-15:
        decay = _resample_decay(decay, ref.tcspc_res, ptu_ref.tcspc_res, ptu_ref.n_bins)
    else:
        decay = _fit_to_grid(decay, ptu_ref.n_bins)
    print(f"  TVB profile from PTU: {total:,.0f} photons")
    return _normalize_profile(decay, normalize)

def tvb_from_decay(decay, n_bins, src_tcspc_res=None, dst_tcspc_res=None, normalize=True):
    decay = np.asarray(decay, dtype=float)
    if (src_tcspc_res is not None and dst_tcspc_res is not None
            and abs(src_tcspc_res - dst_tcspc_res) > 1e-15):
        decay = _resample_decay(decay, src_tcspc_res, dst_tcspc_res, n_bins)
    else:
        decay = _fit_to_grid(decay, n_bins)
    return _normalize_profile(decay, normalize)