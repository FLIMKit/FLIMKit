<!-- Thanks for contributing to FLIMKit. Delete any section that does not apply. -->

## What this changes

<!-- One or two sentences. What does this do, and why? -->

Closes #

## How it was verified

<!-- Be specific and be honest. "Ran the test suite" and "opened one file on my laptop" are
     different claims. If part of this could not be verified, say so under Not verified below. -->

- [ ] `cd flimkit_tests && pytest -m "not requires_data"` passes
- [ ] Added or updated tests covering the change
- [ ] Checked against real data (say which instrument and format below)

Details:

## Not verified

<!-- Anything you could not check: no sample file for a format, no access to the hardware,
     no GPU of that vendor. An unvalidated reader labelled as unvalidated is useful.
     One presented as working is not. Write "nothing" if everything above was checked. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] New or updated file format reader
- [ ] Documentation
- [ ] Build, packaging, or CI
- [ ] Refactor with no behaviour change

## Checklist

- [ ] Branched from `main` with a descriptive branch name
- [ ] The pull request covers one change
- [ ] Style matches the surrounding code (4 spaces, single quotes, no padding around `=`, minimal comments)
- [ ] User-facing changes are reflected in `Docs/documentation.md`, and in `README.md` if it overlaps
- [ ] No data files, credentials, or large binaries committed

### If this adds or changes a format reader

- [ ] Registered in `flimkit/formats/flim_file.py`, with a content sniff if the extension is ambiguous
- [ ] Implements the reader contract in [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-file-format)
- [ ] Added to the supported formats table in `Docs/documentation.md`
- [ ] Specification source recorded in `flimkit/formats/<FORMAT>/NOTICE.md`, crediting anyone who supplied docs or samples
