# Contributing to FLIMKit

Thanks for your interest in FLIMKit. Bug reports, format samples, and pull requests are all welcome.

FLIMKit is maintained by Alex Hunt at the Centre for Inflammation Research, University of Edinburgh. It is MIT licensed, so contributions are accepted under the same terms.

By taking part in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting support

- **Questions about using FLIMKit:** open an [issue](https://github.com/FLIMKit/FLIMKit/issues) and label it `question`.
- **Documentation:** the [wiki](https://github.com/FLIMKit/FLIMKit/wiki) covers installation, workflows, the module reference, and troubleshooting.
- **Email:** Alex Hunt, alexander.hunt@ed.ac.uk. Please prefer issues where possible so answers are searchable by other users.

## Reporting bugs

Open an issue at https://github.com/FLIMKit/FLIMKit/issues and include:

- What you ran (the CLI command, GUI action, or a minimal Python snippet).
- The full traceback, not just the last line.
- Your OS, Python version, and FLIMKit version (`python -c "from flimkit._version import __version__; print(__version__)"`).
- The instrument and file format involved, and the acquisition mode (image or point) if relevant.

Security problems do not go in the issue tracker. See [SECURITY.md](SECURITY.md) for how to report those privately.

FLIM files are often large and sometimes unpublished, so do not attach data you cannot share. A header dump or the file's shape and metadata is usually enough to diagnose a reader problem, and we will ask if we need more.

## Sharing sample files

Sample files are the single most useful contribution to a format reader. Several FLIMKit readers were written from vendor specifications and have never been run against a real acquisition, which is the main source of uncertainty in the project. See the [supported formats table](https://github.com/FLIMKit/FLIMKit/wiki/Supported-Input-Formats) for which readers are validated and which are not.

If you can share a file, a calibration sample with a documented lifetime (a dye with a known tau) is ideal. Best of all is a matched pair: the same sample recorded on two instruments, which allows a bin-for-bin cross-check. Only share data you have the right to share.

## Development setup

```bash
git clone https://github.com/FLIMKit/FLIMKit.git
cd FLIMKit
python install.py --dev
```

`install.py` installs the core requirements, then detects and installs the right GPU backend (MLX on Apple Silicon, CUDA on NVIDIA, ROCm on AMD, or a CPU-only fallback). `--dev` also installs PyInstaller and the test requirements. Python 3.12 or newer is required; official builds use 3.14.

Check the install:

```bash
python validate_installation.py
```

## Running tests

```bash
cd flimkit_tests
pytest -m "not requires_data"
```

This is what CI runs on every push and pull request. Tests marked `requires_data` need real microscopy data that is not in the repository, so they are skipped in CI and will be skipped for you too unless you have the files locally.

Available markers: `unit`, `integration`, `slow`, `requires_data`.

Please add a test for any bug you fix or feature you add. Tests that need real data should be marked `requires_data`; tests that can run on synthetic data should generate it (see `flimkit_tests/mock_data.py` and `flimkit_tests/ptu_writer.py`) so they run in CI.

## Coding style

FLIMKit does not enforce a formatter. Match the surrounding code:

- 4 spaces, single quotes, no alignment padding around `=`.
- Comments are kept minimal. Prefer clear names over explanatory comments.
- Keep functions importable and side-effect free at module level; GUI and heavy optional imports go inside the function that needs them.

## Documentation

The wiki is generated, not edited directly. Edit `Docs/documentation.md` and the `wiki` GitHub Action rebuilds the wiki pages on push to `main`. Each `##` section becomes its own wiki page.

To preview the generated pages locally:

```bash
python Docs/build_wiki.py /tmp/wikitest
```

`README.md` and `Docs/documentation.md` overlap, so if you change something user-facing (a new format, a changed flag) check whether both need updating.

## Adding a file format

Readers live in `flimkit/formats/` and are registered in `flimkit/formats/flim_file.py`. A reader is a class that takes a path and exposes a small contract, so the rest of FLIMKit (fitting, phasor, stitching, ROI, GUI) works unchanged.

Time-domain readers expose `pixel_stack()`, `raw_pixel_stack()`, `summed_decay()`, and the attributes `n_bins`, `tcspc_res`, `time_ns`, `n_x`, `n_y`, `n_channels`, `photon_channel`, and `is_image`. Frequency-domain readers expose `phasor()` returning `(mean, real, imag, frequency_mhz)`. Set `is_image` to `False` when the file has no scan markers so the interface reports a point measurement instead of failing.

If the format is already handled by a maintained third-party reader, prefer delegating to it rather than writing a new decoder. `flimkit/formats/signal.py` and `flimkit/formats/phasor.py` adapt any PhasorPy `signal_from_*` or `phasor_from_*` function to the FLIMKit contract, so many formats are a registry entry and a two-line class.

Add the format to the table in `Docs/documentation.md`, and record where the specification came from in `flimkit/formats/<FORMAT>/NOTICE.md`. If a vendor supplied documentation or sample files, credit them there and in the acknowledgements.

If the extension is ambiguous (`.tif`, `.json`, `.bin`), add a content sniff in `flim_file.py` rather than claiming the extension outright, so FLIMKit does not hijack unrelated files.

## Pull requests

- Branch from `main`, using a descriptive branch name (`feature/...`, `fix/...`).
- Keep the pull request focused on one change.
- Make sure the test suite passes.
- Describe what you changed and how you verified it. If you could not verify part of it (no sample file for a format, no access to the hardware), say so explicitly rather than implying it was tested.

Honest reporting matters more than a clean-looking diff. An unvalidated reader clearly labelled as unvalidated is useful; one presented as working is not.
