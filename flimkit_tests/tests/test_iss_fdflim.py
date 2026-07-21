import struct
import numpy as np
import pytest
from flimkit.formats import FLIMFile, detect_format
from flimkit.formats.flim_file import file_modality
from flimkit.formats.ISS.fdflim import ISSFdFlim, phasor_from_ifli

def _write_ifli(path, phasor, mod_freq, ref_lifetime_ns, ref_phasor,
                n_x, n_y, n_freq=1, n_ch=1, n_z=1, n_ts=1):
    hdr = bytearray(1024)
    struct.pack_into('<12s', hdr, 0, b'VistaFLImage')
    hdr[12] = 16
    fields = struct.pack('<3?', False, False, False)
    fields += struct.pack('<I', 256)
    fields += struct.pack('<I', (1 << n_ch) - 1)
    fields += struct.pack('<B', 0)
    fields += struct.pack('<5H', n_x, n_y, n_z, n_ch, n_ts)
    fields += struct.pack('<6f', 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    fields += struct.pack('<B', 1)
    fields += struct.pack('<4f', 1.0, 1.0, 1.0, 1.0)
    fields += struct.pack('<i', n_freq)
    fields += struct.pack('<f', 0.0)
    fields += struct.pack('<H', 1)
    hdr[13:13 + len(fields)] = fields
    pixel_bytes = np.asarray(phasor, dtype='<f4').tobytes()
    data_off = 1024
    modfreq_off = data_off + len(pixel_bytes)
    reflt_off = modfreq_off + n_freq * 4
    refph_off = reflt_off + n_ch * 3 * 4
    offsets = [data_off, 0, modfreq_off, reflt_off, refph_off,
               0, 0, 0, 0, 0, 0, 0, 0]
    struct.pack_into('<13Q', hdr, 256, *offsets)
    with open(path, 'wb') as fh:
        fh.write(hdr)
        fh.write(pixel_bytes)
        fh.write(np.asarray(mod_freq, '<f4').tobytes())
        fh.write(np.asarray([ref_lifetime_ns] * n_ch, '<f4').tobytes())
        fh.write(np.asarray([1.0] * n_ch, '<f4').tobytes())
        fh.write(np.asarray([0.0] * n_ch, '<f4').tobytes())
        fh.write(np.asarray(ref_phasor, '<f4').tobytes())
    return path

def _simple(path, dc=100.0, gx=0.5, gy=0.3, freq_mhz=20.0, ref_lifetime_ns=0.0):
    n_y, n_x = 2, 3
    phasor = np.zeros((1, 1, 1, n_y, n_x, 1, 3), dtype=np.float32)
    phasor[..., 0] = dc
    phasor[..., 1] = gx
    phasor[..., 2] = gy
    _write_ifli(path, phasor, [freq_mhz], ref_lifetime_ns, [0.0, 0.0, 0.0], n_x, n_y)
    return n_y, n_x

def test_detect_and_dispatch(tmp_path):
    p = tmp_path / 'a.ifli'
    _simple(p)
    assert detect_format(str(p)) == 'iss_fdflim'
    assert file_modality(str(p)) == 'frequency'
    obj = FLIMFile(str(p), verbose=False)
    assert isinstance(obj, ISSFdFlim)
    assert obj.modality == 'frequency'

def test_raw_phasor_roundtrip(tmp_path):
    p = tmp_path / 'a.ifli'
    ny, nx = _simple(p, dc=100.0, gx=0.5, gy=0.3, freq_mhz=20.0)
    iss = ISSFdFlim(str(p), verbose=False)
    mean, real, imag, freq = iss.phasor(calibrate=False)
    assert mean.shape == (ny, nx) and real.shape == (ny, nx)
    assert np.allclose(mean, 100.0)
    assert np.allclose(real, 0.5)
    assert np.allclose(imag, 0.3)
    assert freq == pytest.approx(20.0)

def test_fitting_disabled(tmp_path):
    p = tmp_path / 'a.ifli'
    _simple(p)
    iss = ISSFdFlim(str(p), verbose=False)
    with pytest.raises(ValueError):
        iss.pixel_stack()
    with pytest.raises(ValueError):
        iss.summed_decay()

def test_reference_calibration(tmp_path):
    tau_ns, freq_mhz = 4.0, 20.0
    w = 2.0 * np.pi * (freq_mhz * 1e6)
    wt = w * tau_ns * 1e-9
    gt, st = 1.0 / (1 + wt**2), wt / (1 + wt**2)
    ang, scl = 0.7, 0.6
    mrx = scl * (gt * np.cos(ang) - st * np.sin(ang))
    mry = scl * (gt * np.sin(ang) + st * np.cos(ang))
    p = tmp_path / 'a.ifli'
    phasor = np.zeros((1, 1, 1, 1, 1, 1, 3), dtype=np.float32)
    phasor[..., 0] = 1.0
    phasor[..., 1] = mrx
    phasor[..., 2] = mry
    _write_ifli(p, phasor, [freq_mhz], tau_ns, [1.0, mrx, mry], 1, 1)
    _, real, imag, _ = ISSFdFlim(str(p), verbose=False).phasor(calibrate=True)
    assert real[0, 0] == pytest.approx(gt, abs=1e-4)
    assert imag[0, 0] == pytest.approx(st, abs=1e-4)

def test_process_ifli_returns_phasor_dict(tmp_path):
    from flimkit.phasor.signal import process_ifli
    p = tmp_path / 'a.ifli'
    ny, nx = _simple(p)
    d = process_ifli(str(p))
    assert d['real_cal'].shape == (ny, nx)
    assert d['imag_cal'].shape == (ny, nx)
    assert d['mean'].shape == (ny, nx)
    assert d['frequency'] == pytest.approx(20.0)

def test_phasor_from_ifli_helper(tmp_path):
    p = tmp_path / 'a.ifli'
    _simple(p)
    mean, real, imag, freq = phasor_from_ifli(str(p), calibrate=False)
    assert np.allclose(real, 0.5) and np.allclose(imag, 0.3)
