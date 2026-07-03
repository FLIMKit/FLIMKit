# ISS FastFLIM / Vista support, provenance and acknowledgement

This reader handles two ISS exports: the time-domain time-tag triplet
(`.TAGTIME` + `.TAGCHANNEL` + `.TAGDECAY`) and the plain intensity image (`.ifi`).

## Acknowledgement

Thank you to **ISS, Inc.**, in particular **Dr. Shih-Chu "Jeff" Liao**, for sharing
the FastFLIM / Vista format specifications (FCS, TD-FLIM, FD-FLIM and image), and to
**Anand Yethiraj** (University of Guelph) for making the connection.

The readers in `reader.py` (time-tag) and `image.py` (`.ifi`) are original FLIMKit
code written from those specifications. No ISS source is redistributed here.

## Status: experimental, not yet validated

Both readers have so far only been checked against synthetic files, not against real
ISS acquisitions. The byte order and the frame / line / pixel marker conventions are
assumptions taken from the specifications. Treat ISS results as unverified and
cross-check them. If you have real ISS `.TAGTIME` or `.ifi` data, trying it and
reporting back is very welcome (see issue #19).

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

### Notes / open items for ISS

- Not validated against real files; byte order and marker channel numbers are
  assumptions from the specifications.
- ISS frequency-domain `.ifli` (phasor) is recognised but not decoded yet (issue #19).
