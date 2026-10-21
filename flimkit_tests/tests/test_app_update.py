import sys
import zipfile
import pytest
from flimkit.utils import app_update, update_check

def build(version, dev=False):
    return {'tag': version, 'version': version, 'dev': dev, 'app_name': 'FLIMKit', 'asset': 'FLIMKit-test.zip', 'url': '', 'size': 0}

def test_release_newer_only_when_version_moves():
    assert app_update.is_newer(build('0.13.9'), current='0.13.8') is True
    assert app_update.is_newer(build('0.13.8'), current='0.13.8') is False
    assert app_update.is_newer(build('0.13.7'), current='0.13.8') is False

def test_dev_app_not_offered_the_release_it_was_built_after():
    assert app_update.is_newer(build('0.13.8'), current='0.13.8+dev.2026.09.28') is False
    assert app_update.is_newer(build('0.13.9'), current='0.13.8+dev.2026.09.28') is True

def test_dev_build_compares_dates():
    assert app_update.is_newer(build('2026.09.29', dev=True), current='0.13.8+dev.2026.09.28') is True
    assert app_update.is_newer(build('2026.09.28', dev=True), current='0.13.8+dev.2026.09.28') is False
    assert app_update.is_newer(build('2026.09.28', dev=True), current='0.13.8') is True

def test_install_app_replaces_and_removes_backup(tmp_path):
    target = tmp_path / 'FLIMKit'
    target.write_text('old')
    new = tmp_path / 'new' / 'FLIMKit'
    new.parent.mkdir()
    new.write_text('new')
    app_update.install_app(new, target)
    assert target.read_text() == 'new'
    assert not new.exists()
    assert not (tmp_path / 'FLIMKit.old').exists()

def test_install_app_keep_source_copies_a_bundle(tmp_path):
    target = tmp_path / 'FLIMKit.app'
    (target / 'Contents').mkdir(parents=True)
    (target / 'Contents' / 'marker').write_text('old')
    new = tmp_path / 'dist' / 'FLIMKit.app'
    (new / 'Contents').mkdir(parents=True)
    (new / 'Contents' / 'marker').write_text('new')
    app_update.install_app(new, target, keep_source=True)
    assert (target / 'Contents' / 'marker').read_text() == 'new'
    assert (new / 'Contents' / 'marker').read_text() == 'new'
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith('.flimkit-update-')] == []

def test_install_app_restores_old_copy_when_move_fails(tmp_path, monkeypatch):
    target = tmp_path / 'FLIMKit'
    target.write_text('old')
    new = tmp_path / 'new' / 'FLIMKit'
    new.parent.mkdir()
    new.write_text('new')
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(app_update.shutil, 'move', fail)
    with pytest.raises(OSError):
        app_update.install_app(new, target)
    assert target.read_text() == 'old'

def test_unpack_build_reports_missing_app(tmp_path, monkeypatch):
    monkeypatch.setattr(app_update, 'platform_tag', lambda: 'linux')
    zip_path = tmp_path / 'FLIMKit-linux.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('something-else', 'x')
    with pytest.raises(RuntimeError, match='did not contain FLIMKit'):
        app_update.unpack_build(zip_path, tmp_path / 'out', 'FLIMKit')

def test_unpack_build_makes_linux_binary_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(app_update, 'platform_tag', lambda: 'linux')
    zip_path = tmp_path / 'FLIMKit-linux.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('FLIMKit', 'binary')
    app = app_update.unpack_build(zip_path, tmp_path / 'out', 'FLIMKit')
    assert app.read_text() == 'binary'
    if sys.platform != 'win32':
        assert app.stat().st_mode & 0o111

def test_translocated_app_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match='temporary read-only copy'):
        app_update.check_target('/private/var/folders/x/AppTranslocation/ABC/d/FLIMKit.app')

def test_compiled_app_still_looks_up_the_latest_release(monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    calls = []
    def fake_latest(repo_slug, timeout):
        calls.append(repo_slug)
        return '99.0.0', 'release', None
    monkeypatch.setattr(update_check, '_get_latest_version_from_github', fake_latest)
    status = update_check.check_installation_freshness(do_fetch=False)
    assert calls == ['FLIMKit/FLIMKit']
    assert status['git']['is_repo'] is False
    assert status['release']['latest_version'] == '99.0.0'
    assert status['release']['is_latest'] is False
