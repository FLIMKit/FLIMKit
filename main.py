#!/usr/bin/env python
import os, sys
if getattr(sys, 'frozen', False) and sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
if getattr(sys, 'frozen', False):
    _mpl_cache = os.path.join(sys._MEIPASS, 'mpl-cache')
else:
    _mpl_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpl-cache')
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', _mpl_cache)

from flimkit.cli import main, run

if __name__ == '__main__':
    run()
