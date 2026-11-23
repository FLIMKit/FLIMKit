# Image Cytometry Standard (`.ics`) support, provenance

This reader opens ICS v2 single-file `.ics` files holding a per-pixel TCSPC histogram cube, such as those written by LIFTOFF (KCL's Leica LIF converter). ICS v1 `.ics` + `.ids` pairs are read with the same code. Nothing here has been validated against a real LIFTOFF file.

## Specification sources

- Dean P, Mascio L, Ow D, Sudar D, Mullikin J. Proposed standard for image cytometry data files. Cytometry 11:561-569 (1990).
- SVI libics documentation of the ICS v2 header keywords.
- Format facts supplied by the FLIMKit maintainer for the LIFTOFF variant (layout order `bits y x t`, `parameter scale` carrying the time step in seconds).

No LIFTOFF code (GPL-3) was read, copied, paraphrased or ported. The reader was written from the sources above only.

## Behaviour

- Axes are mapped from `layout order` and `layout sizes`, not hardcoded. `x`, `y` and `t` are required; a `ch`/`c` axis becomes the channel axis; any other axis (`z`, ...) must have size 1 or the reader raises an error.
- The time step comes from `parameter scale` for the `t` axis, converted using `parameter units`.
- Uncompressed data is memory-mapped; gzip data is decompressed in full.
