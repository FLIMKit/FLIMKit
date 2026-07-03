# PicoQuant `.ptu` support and provenance

This reader decodes PicoQuant PTU files (TTTR T3 mode) from PicoHarp, HydraHarp,
TimeHarp 260 and MultiHarp hardware.

## Provenance

The PTU / TTTR format is documented publicly by PicoQuant. The reader in `reader.py`
is original FLIMKit code written from that public documentation; no PicoQuant source
is redistributed here and it does not depend on any PicoQuant software. FLIMKit has
not had direct input from PicoQuant for this reader.

## Format reference (verified against real files)

A PTU file is a tagged header followed by a stream of 32-bit TTTR records.

- **Header**: a magic string, then a list of typed tags (bool, int, float, string,
  date and so on). FLIMKit reads the TCSPC resolution, the sync rate and
  `TTResultFormat_TTTRRecType`, which selects the record layout.
- **T3 records** (32-bit little-endian, decoded with vectorised numpy bit ops):
  - PicoHarp T3 (`0x00010303`): channel = bits 31-28, dtime = bits 27-16,
    nsync = bits 15-0; channel 0xF marks a special record (overflow or marker).
  - HydraHarp / TimeHarp 260 / MultiHarp T3 (the `0x0001030x` / `0x0101030x`
    family): special = bit 31, channel = bits 30-25, dtime = bits 24-10,
    nsync = bits 9-0.
  - nsync overflows are accumulated so the running photon time stays continuous.
- **Image reconstruction**: scan frame / line / pixel markers (special records)
  place each photon in the `(Y, X, H)` decay cube. Files without scan markers
  (point, FCS, TimeHarp 260P point measurements) are read as a decay only.

### Notes / open items for PicoQuant

- T3 mode only. T2 records are raw global timestamps with no per-period decay, so a
  T2 `.ptu` is not decoded yet (see issue #18).
- Older PicoQuant formats (`.pt3`, `.ht3`, `.phu`, `.pt2`) are not read; export `.ptu`
  from the acquisition software instead.
