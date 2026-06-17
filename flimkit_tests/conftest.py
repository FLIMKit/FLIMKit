import sys
from pathlib import Path

_tests_pkg = str(Path(__file__).parent)
_project_root = str(Path(__file__).parent.parent)

for p in (_tests_pkg, _project_root):
    if p not in sys.path:
        sys.path.insert(0, p)
