import time

from flimkit.UI.utils import _Redirect

class _DummyWidget:
    def configure(self, **k): pass
    def insert(self, *a): pass
    def see(self, *a): pass
    def update_idletasks(self): pass

def test_write_no_flush_branch_does_not_crash():
    r = _Redirect(_DummyWidget(), [], root=None, is_stderr=False)
    r._last_flush = time.time()
    r.write('\nZ-stack groups found:')

def test_write_stderr_path_does_not_crash():
    buf = []
    r = _Redirect(_DummyWidget(), buf, root=None, is_stderr=True)
    r._last_flush = time.time()
    r.write('some stderr line')
    assert ''.join(buf) == 'some stderr line'