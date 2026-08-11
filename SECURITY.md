# Security Policy

FLIMKit is a scientific analysis tool. It is maintained by Alex Hunt at the Centre for Inflammation Research, University of Edinburgh, as part of academic research, not as a commercial product with a security team behind it. Reports are handled on a best-effort basis, but they are taken seriously and they will get a reply.

## Supported versions

FLIMKit is pre-1.0 and moves in a single line of development. Only the latest release is supported.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| Anything older | No, please update first |

Check your version with:

```bash
python -c "from flimkit._version import __version__; print(__version__)"
```

If you hit a problem on an older version, please confirm it still happens on the latest release before reporting.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Use one of these instead:

1. GitHub's private vulnerability reporting: go to the [Security tab](https://github.com/flimkit/FLIMKit/security/advisories/new) and open a draft advisory. This is preferred, because it keeps the report, the fix, and the disclosure in one place.
2. Email alexander.hunt@ed.ac.uk with `FLIMKit security` in the subject line.

Please include:

- What the problem is and what an attacker gets out of it.
- The FLIMKit version, your OS, and your Python version.
- Steps to reproduce. If a malformed input file triggers it, a minimal file or a generator script is far more useful than a description.
- Whether you have already disclosed it anywhere else.

### What to expect

- Acknowledgement within 7 days. This is a single-maintainer academic project, so if you have not heard back in that window, send a follow-up rather than assuming it was ignored.
- An assessment within 30 days: whether it is confirmed, what the severity looks like, and a rough fix timeline.
- Credit in the advisory and the release notes, unless you would rather stay anonymous.

Please give a reasonable window to ship a fix before publishing. There is no fixed embargo period, but 90 days is a sensible default and it can be shortened by agreement if the fix lands early.

## Scope

FLIMKit runs locally on data you point it at. The realistic risk is a hostile or corrupt input file, not a remote attacker.

**In scope:**

- Code execution, memory corruption, or path traversal triggered by parsing a FLIM data file (`.ptu`, `.sdt`, `.photons`, `.ifli`, `.tif`, and the rest of the supported formats).
- Code execution or file overwrite triggered by loading a FLIMKit session, project, or config file.
- Anything in the Panel web interface (`flimkit/web/`) that lets a page or a client reach beyond the analysis session: reading arbitrary files, writing outside the project directory, or executing commands.
- Insecure handling of credentials, tokens, or signing material in the build and packaging scripts.
- A dependency vulnerability that FLIMKit actually reaches through its own code paths.

**Out of scope:**

- Crashes, hangs, or unhandled exceptions on malformed files with no path to code execution or data loss. Those are ordinary bugs: please [open an issue](https://github.com/flimkit/FLIMKit/issues), they are still worth fixing.
- Running out of memory on a large file or a large fit. See the documented hardware limits.
- The Panel interface being reachable by other users when you deliberately bind it to a public interface. It is intended for `localhost` and has no authentication layer.
- Vulnerabilities in a dependency that FLIMKit does not call into. Report those upstream.
- Anything requiring an attacker who already has your user account on the machine.

## Notes for users

- FLIMKit parses vendor binary formats, some of them from specifications rather than from validated sample files. Treat data files from people you do not know the same way you would treat any other untrusted binary.
- The web interface has no authentication or authorisation. Do not expose it beyond `localhost` on an untrusted network.
- The maintained third-party readers FLIMKit delegates to (`ptufile`, `sdtfile`, `photonsfile`, `phasorpy`, `lfdfiles`, `tifffile`) have their own security contacts. A bug that reproduces in the upstream reader alone belongs upstream, and it helps to say so in your report if you have already checked.
