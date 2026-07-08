# PicoQuant `.ptu` support and provenance

FLIMKit reads PicoQuant PTU files (TTTR T3 mode) from PicoHarp, HydraHarp, TimeHarp
260 and MultiHarp hardware. Decoding and image reconstruction are done by the
maintained, widely-cited [`ptufile`](https://github.com/cgohlke/ptufile) library
(Christoph Gohlke, BSD-3). FLIMKit's `reader.py` wraps it: it pins the TCSPC bin
grid, integrates frames, selects the photon channel, and exposes the FLIMFile
interface (the `(Y, X, H)` cube, summed decay, sync rate and so on) that the rest of
FLIMKit is built around. The internal FLIMKit representation is a PTU-like array, so
PTU is the format every other reader maps into.

## Provenance

The PTU / TTTR format is documented publicly by PicoQuant; FLIMKit has had no direct
input from PicoQuant. FLIMKit originally shipped its own T3 decoder written from that
public documentation (the PTU reader came first, because the whole `(Y, X, H)`
pipeline is modelled on it and it was where the n-exponential reconvolution was
worked out).

## Why ptufile, and where the original reader went

Before switching, the original decoder was checked against `ptufile` on 32 real
files (Leica FALCON, PicoQuant, Chroma, Zeiss; image and point-mode). Decoding agreed
everywhere: identical decays and per-channel photon counts, and identical
reconstructed images (bit-identical where dense, agreement to the shot-noise limit on
very sparse images). Delegating to `ptufile` puts format correctness on an
established, independently maintained library, and `ptufile` (Cython) is faster and
handles large multi-frame files that were slow or memory-heavy before.

The original hand-rolled decoder is preserved as an independent reference and
cross-check in the `flim-native-decoders` repository, with the comparison script that
reproduces the match.

## Notes

- **Channel numbering** follows `ptufile.active_channels` (0-based enumeration of the
  channels actually present), which can differ from the raw routing-bit value the old
  reader reported for single-detector files. The photon data is the same either way.
- **Point-mode** files (no scan markers: point, FCS, TimeHarp 260P) are detected via
  `ptufile.is_image` and read as a decay only, not a fabricated empty image.
- **`.pck` Check / IRF files** are still read by FLIMKit's own `read_pck`: `ptufile`
  opens them (`PqFile`) but exposes only tags, not the `ChkHistogram` block.
- **T2 mode** is not decoded (raw global timestamps, no per-period decay; see issue
  #18). Older PicoQuant formats (`.pt3`, `.ht3`, `.phu`, `.pt2`) are not read; export
  `.ptu` from the acquisition software.
