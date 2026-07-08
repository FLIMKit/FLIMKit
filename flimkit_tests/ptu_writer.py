import numpy as np
import ptufile

def write_ptu(path, histogram, tcspc_res, frequency, channel=1):
    histogram = np.asarray(histogram)
    if histogram.ndim != 3:
        raise ValueError(f"histogram must be (Y, X, H), got shape {histogram.shape}")
    data = histogram.astype(np.uint16 if int(histogram.max()) < 65536 else np.uint32)
    ptufile.imwrite(str(path), data,
                    global_resolution=1.0 / frequency,
                    tcspc_resolution=tcspc_res)
    return int(data.sum())
