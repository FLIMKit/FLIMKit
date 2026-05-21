#!/usr/bin/env python3
import sys
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


def main():
    print(f"\nFLIMKit Docker Build, Image: {IMAGE}, Version: {VERSION}, Platform: {PLATFORM}")
    if not run(["docker", "buildx", "build",
                "--platform", PLATFORM,
                "-t", f"{IMAGE}:latest",
                "-t", f"{IMAGE}:{VERSION}",
                "--push", "."]):
        sys.exit(1)

if __name__ == "__main__":
    main()
