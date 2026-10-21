import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from flimkit._version import __version__
from flimkit.utils.update_check import _compare_versions, _github_json

REPO = 'FLIMKit/FLIMKit'

def platform_tag():
    if sys.platform == 'darwin':
        return 'macos'
    if sys.platform.startswith('win'):
        return 'windows'
    return 'linux'

def app_file_name(name):
    tag = platform_tag()
    if tag == 'macos':
        return name + '.app'
    if tag == 'windows':
        return name + '.exe'
    return name

def running_app_path():
    if not getattr(sys, 'frozen', False):
        return None
    exe = Path(sys.executable).resolve()
    if platform_tag() == 'macos':
        for parent in exe.parents:
            if parent.suffix == '.app':
                return parent
        return None
    return exe

def is_dev_version(version=__version__):
    return '+dev' in version

def findLatestBuild(dev=None, timeout=10.0):
    if dev is None:
        dev = is_dev_version()
    if dev:
        releases = _github_json(f'https://api.github.com/repos/{REPO}/releases?per_page=20', timeout)
        rel = next((r for r in releases if (r.get('tag_name') or '').startswith('dev-')), None)
        if rel is None:
            raise RuntimeError(f'No dev build is published on github.com/{REPO}.')
        name = 'FLIMKitDEV'
        version = rel['tag_name'][len('dev-'):]
    else:
        rel = _github_json(f'https://api.github.com/repos/{REPO}/releases/latest', timeout)
        name = 'FLIMKit'
        version = (rel.get('tag_name') or '').lstrip('vV')
    asset_name = f'{name}-{platform_tag()}.zip'
    asset = next((a for a in rel.get('assets', []) if a.get('name') == asset_name), None)
    if asset is None:
        raise RuntimeError(f"{rel.get('tag_name')} has no {asset_name} attached, so there is nothing to install for this platform.")
    return {
        'tag': rel.get('tag_name'),
        'version': version,
        'dev': dev,
        'app_name': name,
        'asset': asset_name,
        'url': asset['browser_download_url'],
        'size': asset.get('size') or 0,
    }

def is_newer(build, current=__version__):
    if build['dev']:
        match = re.search(r'\+dev\.(\d{4}\.\d{2}\.\d{2})', current)
        if match is None:
            return True
        return build['version'] > match.group(1)
    return _compare_versions(current, build['version']) == -1

def is_running(target):
    import psutil
    target = Path(target).resolve()
    for proc in psutil.process_iter(['pid', 'exe']):
        if proc.info['pid'] == os.getpid():
            continue
        exe = proc.info.get('exe')
        if not exe:
            continue
        try:
            exe = Path(exe).resolve()
        except OSError:
            continue
        if exe == target or target in exe.parents:
            return True
    return False

def check_target(target):
    target = Path(target)
    if 'AppTranslocation' in str(target):
        raise RuntimeError('macOS is running FLIMKit from a temporary read-only copy, so it cannot be replaced. Move FLIMKit.app into Applications (or any folder), open it from there, and update again.')
    if not os.access(target.parent, os.W_OK):
        raise RuntimeError(f'No permission to write to {target.parent}. Move FLIMKit somewhere you can write to, or update it by hand.')

def download_build(build, dest_dir, progress=None):
    dest = Path(dest_dir) / build['asset']
    req = urllib.request.Request(build['url'], headers={'User-Agent': 'FLIMKit-updater'})
    done = 0
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest, 'wb') as f:
        total = int(resp.headers.get('Content-Length') or build['size'] or 0)
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    if build['size'] and done != build['size']:
        raise RuntimeError(f"Download of {build['asset']} stopped at {done} of {build['size']} bytes.")
    return dest

def unpack_build(zip_path, dest_dir, name):
    dest_dir = Path(dest_dir)
    if platform_tag() == 'macos':
        subprocess.run(['ditto', '-x', '-k', str(zip_path), str(dest_dir)], check=True)
    else:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    app = dest_dir / app_file_name(name)
    if not app.exists():
        raise RuntimeError(f'{Path(zip_path).name} did not contain {app.name}.')
    if platform_tag() == 'linux':
        app.chmod(0o755)
    return app

def remove_path(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()

def remove_leftover_backup(target):
    backup = Path(target).with_name(Path(target).name + '.old')
    try:
        if backup.exists():
            remove_path(backup)
    except OSError:
        pass

def install_app(new_app, target, keep_source=False):
    new_app = Path(new_app)
    target = Path(target)
    check_target(target)
    backup = target.with_name(target.name + '.old')
    remove_leftover_backup(target)
    if backup.exists():
        backup = target.with_name(f'{target.name}.old{os.getpid()}')
    staged = None
    if keep_source:
        staged = Path(tempfile.mkdtemp(dir=target.parent, prefix='.flimkit-update-')) / new_app.name
        if new_app.is_dir():
            shutil.copytree(new_app, staged, symlinks=True)
        else:
            shutil.copy2(new_app, staged)
        new_app = staged
    had_target = target.exists()
    if had_target:
        os.replace(target, backup)
    try:
        shutil.move(str(new_app), str(target))
    except Exception:
        if had_target:
            os.replace(backup, target)
        raise
    finally:
        if staged is not None:
            shutil.rmtree(staged.parent, ignore_errors=True)
    if platform_tag() != 'macos' and target.is_file():
        target.chmod(0o755)
    if had_target:
        try:
            remove_path(backup)
        except OSError:
            pass
    return target

def installBuild(build, target, progress=None):
    target = Path(target)
    check_target(target)
    work = Path(tempfile.mkdtemp(dir=target.parent, prefix='.flimkit-update-'))
    try:
        zip_path = download_build(build, work, progress=progress)
        app = unpack_build(zip_path, work / 'unpacked', build['app_name'])
        return install_app(app, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)

def relaunch(target):
    target = str(target)
    if platform_tag() == 'macos':
        subprocess.Popen(['open', '-n', target])
    elif platform_tag() == 'windows':
        subprocess.Popen([target], creationflags=0x00000008 | 0x00000200, close_fds=True)
    else:
        subprocess.Popen([target], start_new_session=True, close_fds=True)
