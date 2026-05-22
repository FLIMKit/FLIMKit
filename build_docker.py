#!/usr/bin/env python3
import sys
import argparse
import subprocess
from flimkit._version import __version__

IMAGE    = "alex1075/flimkit"
VERSION  = __version__
PLATFORM = "linux/amd64"

def run(cmd):
    print(f"\n▶ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed (exit {e.returncode})")
        return False


def build(dockerfile, tag):
    print(f"\n  [{tag}] {dockerfile}")
    return run(["docker", "buildx", "build",
                "--platform", PLATFORM,
                "-f", dockerfile,
                "-t", f"{IMAGE}:{tag}",
                "-t", f"{IMAGE}:{VERSION}-{tag}",
                "--push", "."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Also build NVIDIA CUDA variant (:cuda tag)")
    parser.add_argument("--rocm", action="store_true", help="Also build AMD ROCm variant (:rocm tag)")
    args = parser.parse_args()

    print(f"\nFLIMKit Docker Build  |  {IMAGE}  |  {VERSION}  |  {PLATFORM}")

    if not build("Dockerfile", "latest"):
        print("\n✗ Build failed. Are you logged in?  docker login -u alex1075")
        sys.exit(1)

    if args.cuda and not build("Dockerfile.cuda", "cuda"):
        sys.exit(1)

    if args.rocm and not build("Dockerfile.rocm", "rocm"):
        sys.exit(1)

    print(f"\n✓ Done!")
    print(f"  {IMAGE}:latest  — CPU")
    if args.cuda:
        print(f"  {IMAGE}:cuda   — NVIDIA  (docker run --gpus all ...)")
    if args.rocm:
        print(f"  {IMAGE}:rocm   — AMD     (docker run --device /dev/kfd --device /dev/dri ...)")

if __name__ == "__main__":
    main()
