import os
import subprocess
import sys
from pathlib import Path

CORE_MODULES = [
    'flimkit',
    'flimkit.configs',
    'flimkit.dialogs',
    'flimkit.interactive',
    'flimkit.project',
    'flimkit.synth',
    'flimkit.formats',
    'flimkit.formats.PTU.stitch',
    'flimkit.FLIM.fitters',
    'flimkit.FLIM.irf_tools',
    'flimkit.phasor',
    'flimkit.plugins',
    'flimkit.utils.config_snapshot',
    'flimkit.utils.display',
    'flimkit.utils.roi',
    'flimkit.utils.session',
]

PURITY = '''
import sys
import importlib

for name in {modules!r}:
    importlib.import_module(name)

leaked = sorted(m for m in sys.modules if m == 'tkinter' or m.startswith('tkinter.') or m.startswith('flimkit.UI'))
if leaked:
    raise SystemExit('headless core pulled in the desktop frontend: ' + ', '.join(leaked))
'''

MISSING_TK = '''
import sys
import importlib
import importlib.abc

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'tkinter' or name.startswith('tkinter.'):
            raise ImportError('no tkinter on this machine')
        return None

sys.meta_path.insert(0, Block())
for name in {modules!r}:
    importlib.import_module(name)
'''

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def probe_env():
    existing = os.environ.get('PYTHONPATH', '')
    path = REPO_ROOT + (os.pathsep + existing if existing else '')
    return dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=path)


def run_probe(source):
    env = probe_env()
    return subprocess.run(
        [sys.executable, '-c', source.format(modules=CORE_MODULES)],
        capture_output=True, text=True, env=env)

def test_core_does_not_import_the_desktop_frontend():
    proc = run_probe(PURITY)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_core_imports_on_a_machine_without_tkinter():
    proc = run_probe(MISSING_TK)
    assert proc.returncode == 0, proc.stderr
