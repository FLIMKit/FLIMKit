#!/usr/bin/env python3
import sys
import subprocess
from flimkit._version import __version__

IMAGE = 'alex1075/flimkit'
VERSION = __version__
PLATFORM = 'linux/amd64'

VARIANTS = [
    ('Dockerfile', 'latest', 'CPU'),
    ('Dockerfile.cuda', 'cuda', 'NVIDIA  (docker run --gpus all ...)'),
    ('Dockerfile.rocm', 'rocm', 'AMD     (docker run --device /dev/kfd --device /dev/dri ...)'),
]

def run(cmd):
    print(f'\n▶ {" ".join(cmd)}')
    try:
        subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f'✗ Failed (exit {e.returncode})')
        return False

def build(dockerfile, tag):
    print(f'\n  [{tag}] {dockerfile}')
    return run(['docker', 'buildx', 'build',
                '--platform', PLATFORM,
                '-f', dockerfile,
                '-t', f'{IMAGE}:{tag}',
                '-t', f'{IMAGE}:{VERSION}-{tag}',
                '--push', '.'])

def main():
    print(f'\nFLIMKit Docker Build  |  {IMAGE}  |  {VERSION}  |  {PLATFORM}')
    for dockerfile, tag, _ in VARIANTS:
        if not build(dockerfile, tag):
            print('\n✗ Build failed. Are you logged in?  docker login -u alex1075')
            sys.exit(1)
    print('\n✓ Done!')
    for _, tag, desc in VARIANTS:
        print(f'  {IMAGE}:{tag}  — {desc}')

if __name__ == '__main__':
    main()
