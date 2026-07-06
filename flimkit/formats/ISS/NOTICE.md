# ISS FastFLIM / Vista support, provenance and acknowledgement

This package handles three ISS exports: the time-domain time-tag triplet
(`.TAGTIME` + `.TAGCHANNEL` + `.TAGDECAY`), the plain intensity image (`.ifi`), and
the frequency-domain FD-FLIM lifetime image (`.ifli`).

## Acknowledgement

Thank you to **ISS, Inc.**, in particular **Dr. Shih-Chu "Jeff" Liao**, for sharing
the FastFLIM / Vista format specifications (FCS, TD-FLIM, FD-FLIM and image), and to
**Anand Yethiraj** (University of Guelph) for making the connection.

The readers in `reader.py` (time-tag), `image.py` (`.ifi`) and `fdflim.py` (`.ifli`)
are original FLIMKit code written from those specifications. No ISS source is
redistributed here.

## Status: experimental, not yet validated

The readers have so far only been checked against synthetic files, not against real
ISS acquisitions. Byte order, the frame / line / pixel marker conventions, and the
FD-FLIM modulation-frequency units are assumptions taken from the specifications.
Treat ISS results as unverified and cross-check them. If you have real ISS
`.TAGTIME`, `.ifi` or `.ifli` data, trying it and reporting back is very welcome
(see issue #19).

## Format reference (from the ISS specifications)

- **Time-tag triplet** (`.TAGTIME` / `.TAGCHANNEL` / `.TAGDECAY`): the three files
  are read together (point the loader at any one of them or their shared basename).
  Each file starts with a 4-byte excitation period in picoseconds, then a stream of
  per-event records: `.TAGTIME` = arrival time (int64 ps), `.TAGCHANNEL` = detector
  or marker channel (int32), `.TAGDECAY` = TCSPC bin (int32). Detector channels are
  1..4; ISS records explicit frame (channel 5), line (6) and pixel (7) markers, so
  the per-pixel image is reconstructed exactly from the markers rather than inferred.
- **`.ifi` intensity image** (`VISTAIMAGE` header): float pixels per channel and
  frame, no lifetime data, so it loads as an intensity image only (no fitting or
  phasor).
- **`.ifli` FD-FLIM lifetime image** (`VistaFLImage` header, 1024-byte header): the
  frequency-domain export. After the header, each pixel stores a `DcPhasor`
  (`DC`, `PhasorX`, `PhasorY` as float32) at every modulation frequency, looped as
  time series / channels / Z / frame. `PhasorX` and `PhasorY` (normalised by `DC`)
  are the phasor coordinates, so the reader loads them straight into phasor analysis
  with fitting disabled, applying the file's reference-sample calibration
  (`RefDcPhasor` + `RefLifetime`) at each frequency.

### Notes / open items for ISS

- Not validated against real files; byte order, marker channel numbers and the
  `.ifli` modulation-frequency units (Hz vs MHz) are assumptions from the specs.
- FD-FLIM `.ifli` decoding and its reference calibration are written from the spec
  and checked on synthetic files, but need a real `.ifli` to confirm (issue #19).
