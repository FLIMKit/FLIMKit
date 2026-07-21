import os

DEFAULT_BACKEND = 'TkAgg'
ENV_VAR = 'FLIMKIT_MPL_BACKEND'

_selected = None

def select_backend(name=None):
    global _selected
    import matplotlib
    if _selected is not None and name is None:
        return _selected
    requested = name or os.environ.get(ENV_VAR) or DEFAULT_BACKEND
    try:
        matplotlib.use(requested)
    except Exception:
        matplotlib.use(DEFAULT_BACKEND)
    _selected = matplotlib.get_backend()
    return _selected

def current_backend():
    return _selected
