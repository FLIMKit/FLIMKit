#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parent
sys.path.insert(0, str(repo))

def run_git(*args):
    result = subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True, check=False)
    return result.returncode == 0, (result.stdout + result.stderr).strip()

def read_version():
    text = (repo / 'flimkit' / '_version.py').read_text()
    match = re.search(r"__version__ = '([^']+)'", text)
    if match:
        return match.group(1)
    return 'unknown'

def built_app_path():
    from flimkit.utils.app_update import app_file_name
    name = 'FLIMKitDEV' if '+dev' in read_version() else 'FLIMKit'
    return repo / 'dist' / app_file_name(name)

def installed_copy():
    code = 'import flimkit, os; print(os.path.dirname(os.path.realpath(flimkit.__file__)))'
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run([sys.executable, '-c', code], cwd=tmp, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())

def pull_code(dry_run):
    ok, out = run_git('rev-parse', '--show-toplevel')
    if not ok:
        print(f'{repo} is not a git checkout, so there is nothing to pull. Clone https://github.com/FLIMKit/FLIMKit.git or download the app from the Releases page.')
        return 'error'
    ok, branch = run_git('rev-parse', '--abbrev-ref', 'HEAD')
    ok, upstream = run_git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}')
    if not ok:
        print(f'Branch {branch} has no upstream to pull from. Set one with: git branch --set-upstream-to origin/{branch}')
        return 'error'
    ok, dirty = run_git('status', '--porcelain', '--untracked-files=no')
    if dirty:
        print('These files have changes that a pull could overwrite. Commit or stash them first, then run this again:')
        print(dirty)
        return 'error'
    print(f'FLIMKit {read_version()} on {branch}, pulling from {upstream}')
    ok, out = run_git('fetch', '--quiet')
    if not ok:
        print(f'git fetch failed: {out}')
        return 'error'
    ok, incoming = run_git('log', '--oneline', f'HEAD..{upstream}')
    if not incoming:
        print('The code is already up to date.')
        return 'current'
    print(f'{len(incoming.splitlines())} new commit(s):')
    print(incoming)
    if dry_run:
        print('Dry run: would run git pull --ff-only')
        return 'updated'
    ok, out = run_git('pull', '--ff-only')
    if not ok:
        print(f'git pull --ff-only failed, so nothing was changed. Your branch and {upstream} have both moved on; merge or rebase by hand.')
        print(out)
        return 'error'
    print(f'Now at FLIMKit {read_version()}')
    return 'updated'

def install_requirements(dev, dry_run):
    install_cmd = [sys.executable, str(repo / 'install.py')]
    if dev:
        install_cmd.append('--dev')
    if dry_run:
        print(f"Dry run: would run {' '.join(install_cmd)}")
        return True
    if subprocess.run(install_cmd, cwd=repo, check=False).returncode != 0:
        print('install.py failed. The code is updated but some requirements may be missing; fix the error above and run python install.py.')
        return False
    copy = installed_copy()
    if copy is not None and repo not in copy.parents:
        print(f'An installed copy of flimkit at {copy} would be imported instead of this checkout. Reinstalling it from the checkout.')
        if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', str(repo)], check=False).returncode != 0:
            print(f'Reinstalling failed. Run: {sys.executable} -m pip install --upgrade {repo}')
            return False
    return True

def refuse_if_running(*paths):
    from flimkit.utils.app_update import is_running
    for path in paths:
        if path is not None and Path(path).exists() and is_running(path):
            print(f'{path} is running. Quit FLIMKit and run this again.')
            return True
    return False

def rebuildApp(app_path, dry_run):
    from flimkit.utils.app_update import install_app
    built = built_app_path()
    if subprocess.run([sys.executable, '-c', 'import PyInstaller'], capture_output=True, check=False).returncode != 0:
        print('PyInstaller is not installed in this environment, so the app cannot be rebuilt. Run python install.py --dev first.')
        return False
    if refuse_if_running(built, app_path):
        return False
    if dry_run:
        print(f'Dry run: would run build_and_sign.py to rebuild {built}')
        if app_path is not None:
            print(f'Dry run: would copy it to {app_path}')
        return True
    print(f'Rebuilding {built}')
    if subprocess.run([sys.executable, str(repo / 'build_and_sign.py')], cwd=repo, check=False).returncode != 0:
        print('build_and_sign.py failed, see above. The code and requirements are updated; only the app is not.')
        return False
    if not built.exists():
        print(f'build_and_sign.py finished but {built} is not there.')
        return False
    if app_path is not None:
        try:
            install_app(built, app_path, keep_source=True)
        except Exception as exc:
            print(f'Could not replace {app_path}: {exc}')
            return False
        print(f'Replaced {app_path} with the new build')
    return True

def installRelease(app_path, dev, dry_run):
    from flimkit.utils.app_update import findLatestBuild, installBuild
    try:
        build = findLatestBuild(dev=dev)
    except Exception as exc:
        print(f'Could not find the latest build on GitHub: {exc}')
        return False
    print(f"Latest {'dev build' if dev else 'release'}: {build['tag']}, {build['asset']} ({build['size'] / 1e6:.0f} MB)")
    if refuse_if_running(app_path):
        return False
    if dry_run:
        print(f'Dry run: would download it and replace {app_path}')
        return True
    last = [0]
    def show(done, total):
        if total and done - last[0] >= total / 10:
            last[0] = done
            print(f'  {done / 1e6:.0f} of {total / 1e6:.0f} MB', flush=True)
    try:
        installBuild(build, app_path, progress=show)
    except Exception as exc:
        print(f"Could not install {build['tag']}: {exc}")
        return False
    print(f"Replaced {app_path} with {build['tag']}")
    return True

def updateInstall(dev=False, dry_run=False, force=False, rebuild=None, app_path=None, release=None):
    if app_path is not None:
        app_path = Path(app_path).expanduser().resolve()
    if release is not None:
        if app_path is None:
            print('--release and --dev-release need --app-path, the app to replace.')
            return 1
        return 0 if installRelease(app_path, release == 'dev', dry_run) else 1
    state = pull_code(dry_run)
    if state == 'error':
        return 1
    changed = state == 'updated' or force
    if changed and not install_requirements(dev, dry_run):
        return 1
    if rebuild is None:
        rebuild = changed and built_app_path().exists()
        if rebuild:
            print(f'Found a local build at {built_app_path()}, so rebuilding it. Use --no-rebuild to skip.')
    if rebuild and not rebuildApp(app_path, dry_run):
        return 1
    if not changed and not rebuild:
        print('Nothing to do. --force reruns install.py, --rebuild rebuilds the app.')
        return 0
    print('Update complete. Run python validate_installation.py to check it.')
    return 0

def main():
    parser = argparse.ArgumentParser(description='Update a FLIMKit checkout: git pull, rerun install.py, and rebuild a locally built app')
    parser.add_argument('--dev', action='store_true', help='pass --dev to install.py (PyInstaller and test requirements)')
    parser.add_argument('--dry-run', action='store_true', help='show what would happen without changing anything')
    parser.add_argument('--force', action='store_true', help='rerun install.py even when there is nothing to pull')
    parser.add_argument('--rebuild', dest='rebuild', action='store_true', default=None, help='rebuild the app with build_and_sign.py (the default when dist/ already has a build and the code changed)')
    parser.add_argument('--no-rebuild', dest='rebuild', action='store_false', help='never rebuild the app')
    parser.add_argument('--app-path', help='after a rebuild, copy the new app over this one, e.g. /Applications/FLIMKit.app')
    channel = parser.add_mutually_exclusive_group()
    channel.add_argument('--release', dest='release', action='store_const', const='release', help='skip the clone and replace --app-path with the latest release from GitHub')
    channel.add_argument('--dev-release', dest='release', action='store_const', const='dev', help='the same, with the latest FLIMKitDEV build')
    args = parser.parse_args()
    return updateInstall(dev=args.dev, dry_run=args.dry_run, force=args.force, rebuild=args.rebuild, app_path=args.app_path, release=args.release)

if __name__ == '__main__':
    sys.exit(main())
