# Becker & Hickl `.sdt` support — provenance and acknowledgement

This reader decodes Becker & Hickl SPC FLIM data files (`.sdt`).

## Acknowledgement

Thank you to **Becker & Hickl GmbH** for supporting FLIMKit, in particular
**Dr. Jens Balke** and **Enzo Marscheck**, who provided the official SPCM
file-structure documentation (`SPC_data_file_structure.h`) and sample data.

The decoder in `decode.py` is original FLIMKit code written from that
documentation, the same way the PicoQuant reader was written from PicoQuant's
format docs. Becker & Hickl's own Python importer (destined for the MIT-licensed
`bhpy` package) was used only as a secondary cross-check. No Becker & Hickl source
is redistributed here.

## Format reference (verified against sample files)

`.sdt` layout (little-endian, packed; see SPCM `SPC_data_file_structure.h`):

- **File header** (42 bytes): `revision` (low 4 bits = software revision, bits 11-4
  = module type code), offsets/lengths for the info / setup / measurement-description
  / data-block sections, `header_valid` == 0x5555, and `reserved1` carrying the real
  `no_of_data_blocks` when the short field is 0x7fff.
- **Identification** (`*IDENTIFICATION` ASCII block) carries the ID string, title,
  version (file type: 1 non-FIFO, 2 FIFO, 3 FIFO-image) and ADC-bit revision.
- **Measurement description** (`MeasureInfo`): the core fields FLIMKit uses are
  `meas_mode`, `tac_r`, `tac_g`, `adc_re` (0 means 65536), `scan_x/scan_y`,
  `mod_type`, `image_x/image_y`, and `StopInfo.min_sync_rate/max_sync_rate`. The
  block length is 512 bytes on older files (no extension) and 2048 with the
  extension; the core fields sit at the same offsets in both.
- **Data block headers** (revision 15): 22 bytes, with 40-bit `data_offs` /
  `next_block_offs` (the high byte in `*_offs_ext`). `block_type` encodes the
  creation mode (bits 0-3), content type (bits 4-7, e.g. IMG=0x60, IMG_INT=0xA0),
  data type (bits 8-11, USHORT/ULONG/DBL) and compression (bit 12 = zip, bit 14 =
  LZ4). `block_length` is the true uncompressed size in bytes.

### Notes / open items for B&H

- There is an undocumented 4-byte field between `FCSInfo` and `image_x` in
  `MeasureInfo` (observed value 250, a macro-time clock in 0.1 ns units). Skipping it
  makes `image_x`/`image_y` agree exactly with `block_length` across all sample
  files. Confirm what this field is.
- Image data layout (power-of-two padding), `tac_r` units, and routing-channel ↔
  block mapping are pending confirmation; see the project plan's question list.
