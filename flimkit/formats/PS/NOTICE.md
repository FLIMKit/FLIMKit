# Photonscore `.photons` support, provenance and acknowledgement

This reader decodes Photonscore LINCam `.photons` files (the "D7" container), the
position-sensitive photon-counting format written by Photonscore's LINCam systems.

## Acknowledgement

Thank you to **Photonscore GmbH** for supporting FLIMKit: they shared the LINCam
SDK and a sample `.photons` file, and open-sourced the D7 storage format at
[github.com/photonscore/d7](https://github.com/photonscore/d7) (Apache-2.0). At
Photonscore's request the credit here is to the company rather than to an
individual.

The decoder in `decode.py` is original FLIMKit code. It was first worked out from
the SDK and the sample file, then checked field for field against the public D7
specification. No Photonscore source is redistributed here. The reader is pure
Python (numpy, with an optional numba path) and needs none of Photonscore's native
libraries.

## Format reference (verified bit-exact against the SDK)

A `.photons` file is a stream of 16384-byte pages. Each page starts with a 2-byte
marker that is not part of the logical stream; with the markers removed the stream
is a protobuf `Header`, a run of `Data` blocks, a global `Index` and an `Epilogue`.

- **Header**: a protobuf message ("D7 Photons Data" signature, version, page size
  and a table of datasets). Each dataset has a name, a type code (0-9 = int8/16/32/64,
  uint8/16/32/64, float, double) and a size.
- **Data block**: a protobuf message holding one dataset's values as a `seed` plus
  the cumulative sum of zigzag-varint deltas (`integers` for 32-bit, `longs` for
  64-bit). Blocks hold about 8192 values each.
- **Index / Epilogue**: the `Epilogue` at the end of the file points at the global
  `Index`, which lists the file offset of every data block plus the file attributes.
  Index checkpoints are also written between the data blocks, so decoding follows the
  index offsets rather than walking the stream.
- **Datasets** used for FLIM: `/photons/x`, `/photons/y` (position, 0..2^PositionBits),
  `/photons/dt` (TCSPC micro-time, 0..2^TacBits) and `/photons/ms` (millisecond
  macro-time markers).

### Calibration and timing

- `dt` to time comes from the `/photons/TacChannel` attribute (picoseconds per raw
  `dt` unit). Pass `period_ns=` to override it.
- Newer LINCam systems record two TDCs: laser reference `/start/time` and photon
  `/stop/time` as int64 picoseconds. When both are present the reader uses
  `dt = stop - start`, otherwise it falls back to `/photons/dt`.

### Notes / open items for Photonscore

- The dual-TDC (`/start/time` / `/stop/time`) path is written from the format
  description but not yet checked against a real new-format acquisition.
- Raw `float` / `double` / `bytes` datasets (D7 `Data` fields 6/7/8) are not decoded;
  current LINCam FLIM files only use the integer delta path.
