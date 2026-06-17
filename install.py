#!/usr/bin/env python3
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

class _C:
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RED    = "\033[91m"
    END    = "\033[0m"

def _hdr(msg):
    print(f"\n{_C.BOLD}{_C.CYAN}{'─'*60}{_C.END}")
    print(f"{_C.BOLD}{_C.CYAN}  {msg}{_C.END}")
    print(f"{_C.BOLD}{_C.CYAN}{'─'*60}{_C.END}")

def _ok(msg)   : print(f"  {_C.GREEN}✓{_C.END}  {msg}")
def _info(msg) : print(f"  {_C.YELLOW}→{_C.END}  {msg}")
def _err(msg)  : print(f"  {_C.RED}✗{_C.END}  {msg}")

def _run(cmd, dry_run):
    """Print and optionally execute *cmd*.  Returns True on success."""
    display = " ".join(cmd)
    _info(display)
    if dry_run:
        return True
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        _err(f"Command failed (exit {result.returncode})")
        return False
    return True

def _pip(*args, dry_run=False):
    return _run([sys.executable, "-m", "pip", "install", "--upgrade", *args],
                dry_run=dry_run)

def _detect_gpu():
    """Return a short token describing the best GPU target.

    Tokens:
        mlx       — Apple Silicon macOS (MLX preferred)
        mps       — macOS without Apple Silicon (Intel Mac, MPS unavailable
                    but return mps so torch CPU wheel still gets installed)
        cuda      — NVIDIA CUDA (Linux / Windows)
        rocm      — AMD ROCm (Linux)
        cpu       — no GPU or unrecognised
    """
    os_name = sys.platform
    machine = platform.machine()

    if os_name == "darwin":
        if machine == "arm64":
            return "mlx"
        return "mps"

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=False, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return "cuda"
        except Exception:
            pass

    if shutil.which("rocminfo") or Path("/opt/rocm").exists():
        return "rocm"

    if os_name == "linux" and shutil.which("lspci"):
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, check=False, timeout=5)
            out = result.stdout.lower()
            if "nvidia" in out:
                return "cuda"
            if "amd" in out or "radeon" in out:
                return "rocm"
        except Exception:
            pass

    return "cpu"

_TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu126",
    "rocm": "https://download.pytorch.org/whl/rocm6.2",
    "cpu":  "https://download.pytorch.org/whl/cpu",
}

def _install_gpu(token, dry_run):
    _hdr(f"Installing GPU libraries  [{token.upper()}]")

    if token == "mlx":
        # Apple Silicon: install MLX (zero-copy Metal) + PyTorch CPU/MPS wheel
        _info("Installing Apple MLX (Metal GPU)…")
        ok = _pip("mlx", dry_run=dry_run)
        if ok:
            _ok("mlx installed")
        _info("Installing PyTorch (MPS fallback)…")
        ok = _pip("torch", "--index-url",
                  "https://download.pytorch.org/whl/cpu",
                  dry_run=dry_run)
        if ok:
            _ok("torch (MPS) installed")
    elif token == "mps":
        # Intel Mac: only CPU torch; MPS requires Apple Silicon
        _info("Intel Mac detected — installing PyTorch CPU wheel.")
        _info("(MPS backend requires Apple Silicon; GPU acceleration unavailable.)")
        ok = _pip("torch", "--index-url",
                  "https://download.pytorch.org/whl/cpu",
                  dry_run=dry_run)
        if ok:
            _ok("torch (cpu) installed")
    elif token in ("cuda", "rocm"):
        index = _TORCH_INDEX[token]
        _info(f"Installing PyTorch for {token.upper()} from {index} …")
        ok = _pip("torch", "--index-url", index, dry_run=dry_run)
        if ok:
            _ok(f"torch ({token}) installed")
    else:
        _info("No GPU detected — installing CPU-only PyTorch…")
        ok = _pip("torch", "--index-url", _TORCH_INDEX["cpu"], dry_run=dry_run)
        if ok:
            _ok("torch (cpu) installed")

def main():
    parser = argparse.ArgumentParser(
        description="FLIMKit installer — installs core + platform-appropriate GPU deps")
    parser.add_argument("--dev",     action="store_true",
                        help="Also install PyInstaller and test requirements")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print install commands without executing them")
    args = parser.parse_args()
    print(f"\n{_C.BOLD}FLIMKit Installer{_C.END}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Platform : {sys.platform}  ({platform.machine()})")
    print(f"  Dry-run  : {args.dry_run}")
    print(f"  Dev deps : {'install' if args.dev else 'skip (use --dev to include)'}")
    req_file = Path(__file__).parent / "requirements.txt"
    _hdr("Installing core requirements")
    filtered_lines = []
    for line in req_file.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            filtered_lines.append(stripped)
    import tempfile, os
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(filtered_lines))
        tmp_path = tmp.name
    try:
        ok = _run([sys.executable, "-m", "pip", "install", "--upgrade",
                   "-r", tmp_path], dry_run=args.dry_run)
    finally:
        os.unlink(tmp_path)
    if not ok:
        _err("Core requirements failed — aborting.")
        return 1
    _ok("Core requirements installed")

    if args.dev:
        _hdr("Installing PyInstaller")
        ok = _pip("pyinstaller", dry_run=args.dry_run)
        if ok:
            _ok("PyInstaller installed")
        else:
            _err("PyInstaller install failed — continuing.")

        test_req = Path(__file__).parent / "flimkit_tests" / "requirements_test.txt"
        if test_req.exists():
            _hdr("Installing test requirements")
            ok = _run([sys.executable, "-m", "pip", "install", "--upgrade",
                       "-r", str(test_req)], dry_run=args.dry_run)
            if ok:
                _ok("Test requirements installed")
            else:
                _err("Test requirements install failed — continuing.")
        else:
            _info(f"Test requirements file not found: {test_req}")

    token = _detect_gpu()
    _info(f"Detected GPU target: {token.upper()}")
    _install_gpu(token, dry_run=args.dry_run)
    _hdr("Installation complete")
    _ok("Run  python validate_installation.py  to verify the setup.")
    if token != "cpu":
        _ok(f"GPU backend will be auto-detected at runtime via  flimkit.GPU.get_backend().")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())