#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parent

def run_git(*args):
    result = subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True, check=False)
    return result.returncode == 0, (result.stdout + result.stderr).strip()

def read_version():
    text = (repo / 'flimkit' / '_version.py').read_text()
    match = re.search(r"__version__ = '([^']+)'", text)
    if match:
        return match.group(1)
    return 'unknown'

def installed_copy():
    code = 'import flimkit, os; print(os.path.dirname(os.path.realpath(flimkit.__file__)))'
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run([sys.executable, '-c', code], cwd=tmp, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())

def updateInstall(dev=False, dry_run=False, force=False):
    ok, out = run_git('rev-parse', '--show-toplevel')
    if not ok:
        print(f'{repo} is not a git checkout, so there is nothing to pull. Clone https://github.com/FLIMKit/FLIMKit.git or download the app from the Releases page.')
        return 1
    ok, branch = run_git('rev-parse', '--abbrev-ref', 'HEAD')
    ok, upstream = run_git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}')
    if not ok:
        print(f'Branch {branch} has no upstream to pull from. Set one with: git branch --set-upstream-to origin/{branch}')
        return 1
    ok, dirty = run_git('status', '--porcelain', '--untracked-files=no')
    if dirty:
        print('These files have changes that a pull could overwrite. Commit or stash them first, then run this again:')
        print(dirty)
        return 1
    print(f'FLIMKit {read_version()} on {branch}, pulling from {upstream}')
    ok, out = run_git('fetch', '--quiet')
    if not ok:
        print(f'git fetch failed: {out}')
        return 1
    ok, incoming = run_git('log', '--oneline', f'HEAD..{upstream}')
    if not incoming and not force:
        print('Already up to date. Use --force to reinstall the requirements anyway.')
        return 0
    if incoming:
        print(f'{len(incoming.splitlines())} new commit(s):')
        print(incoming)
    if dry_run:
        print(f'Dry run: would run git pull --ff-only, then {Path(sys.executable).name} install.py')
        return 0
    ok, out = run_git('pull', '--ff-only')
    if not ok:
        print(f'git pull --ff-only failed, so nothing was changed. Your branch and {upstream} have both moved on; merge or rebase by hand.')
        print(out)
        return 1
    print(f'Now at FLIMKit {read_version()}')
    install_cmd = [sys.executable, str(repo / 'install.py')]
    if dev:
        install_cmd.append('--dev')
    if subprocess.run(install_cmd, cwd=repo, check=False).returncode != 0:
        print('install.py failed. The code is updated but some requirements may be missing; fix the error above and run python install.py.')
        return 1
    copy = installed_copy()
    if copy is not None and repo not in copy.parents:
        print(f'An installed copy of flimkit at {copy} would be imported instead of this checkout. Reinstalling it from the checkout.')
        if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', str(repo)], check=False).returncode != 0:
            print(f'Reinstalling failed. Run: {sys.executable} -m pip install --upgrade {repo}')
            return 1
    print('Update complete. Run python validate_installation.py to check it.')
    return 0

def main():
    parser = argparse.ArgumentParser(description='Update a FLIMKit checkout: git pull, then install.py for the requirements and GPU backend')
    parser.add_argument('--dev', action='store_true', help='pass --dev to install.py (PyInstaller and test requirements)')
    parser.add_argument('--dry-run', action='store_true', help='show the incoming commits without pulling or installing')
    parser.add_argument('--force', action='store_true', help='rerun install.py even when there is nothing to pull')
    args = parser.parse_args()
    return updateInstall(dev=args.dev, dry_run=args.dry_run, force=args.force)

if __name__ == '__main__':
    sys.exit(main())
