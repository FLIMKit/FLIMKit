# Becker & Hickl `.sdt` support, provenance and acknowledgement

This reader decodes Becker & Hickl SPC FLIM data files (`.sdt`). Reading is done by
the maintained, widely-cited [`sdtfile`](https://github.com/cgohlke/sdtfile) library
(Christoph Gohlke, BSD-3). FLIMKit's `reader.py` wraps it: it selects the brightest
photon channel, reshapes each block into the `(Y, X, H)` decay cube, computes the
TCSPC resolution and sync rate, and exposes the FLIMFile interface the rest of
FLIMKit uses.

## Acknowledgement

Thank you to **Becker & Hickl GmbH** for supporting FLIMKit, in particular
**Dr. Jens Balke** and **Enzo Marscheck**, who provided the official SPCM
file-structure documentation (`SPC_data_file_structure.h`, kept here for reference)
and sample data.

## Why sdtfile, and where the original reader went

FLIMKit originally shipped its own `.sdt` decoder, written from the SPCM
documentation. Before switching, that decoder was checked against `sdtfile` on real
acquisitions: the decoded cubes were **bit-identical** (element for element, both
channels, across 1024x1024 and 512x512 files from SPC-150NX and HPM detectors), and
`tcspc_res` matched exactly. Delegating to `sdtfile` means format correctness now
rests on an established, independently maintained library, which is one less thing
FLIMKit has to prove.

The original hand-rolled decoder is preserved as an independent reference and
cross-check in the `flim-native-decoders` repository, together with the comparison
script that reproduces the bit-for-bit match.

## One thing the wrapper still does itself

`sdtfile` reshapes a block only when its stored size equals `image_x * image_y *
adc_re` (or a scan-dimension variant); otherwise it returns `(pixels, adc_re)`. It
does not undo Becker & Hickl's power-of-two image padding. FLIMKit keeps a small
pad-and-crop step (`_reshape_cube`) so non-power-of-two images stored with padding
still come back as the correct `(Y, X, H)` cube. All real sample files are already
power-of-two, so this only matters for non-standard image sizes.

## Format reference (what sdtfile parses)

`.sdt` layout (little-endian, packed; see SPCM `SPC_data_file_structure.h`):

- **File header**: `revision` (low 4 bits = software revision, bits 11-4 = module
  type code), section offsets/lengths, `header_valid` == 0x5555.
- **Identification** (`*IDENTIFICATION` ASCII block): ID string, title, date, time.
- **Measurement description** (`MeasureInfo`): the fields FLIMKit reads are
  `meas_mode`, `tac_r`, `tac_g`, `adc_re` (0 means 65536), `image_x/image_y`,
  `mod_type`, and `StopInfo.min_sync_rate/max_sync_rate`.
- **Data blocks**: `block_type` encodes the content type (bits 4-7, e.g. IMG=0x60,
  IMG_INT=0xA0); a decay-cube block pairs with its intensity block by
  `meas_desc_block_no`. FLIMKit computes `tcspc_res = tac_r / (tac_g * adc_re)` and
  uses bin centres for the time axis.

### Resolved open items

Earlier notes flagged an undocumented 4-byte field before `image_x`, the power-of-two
padding, `tac_r` units, and the routing-channel to block mapping as pending
confirmation. Reading through `sdtfile` and matching it bit-for-bit on the real
samples resolves these: the layout and calibration FLIMKit used were correct.
