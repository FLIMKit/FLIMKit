_provider = None

def set_dialog_provider(fn):
    global _provider
    _provider = fn

def get_dialog_provider():
    return _provider

def _tk_save_path(title, default_name, defaultextension, filetypes):
    import tkinter as tk
    from tkinter import filedialog
    existing = tk._default_root
    if existing is not None:
        parent = existing
        need_destroy = False
    else:
        parent = tk.Tk()
        parent.withdraw()
        need_destroy = True
    parent.attributes('-topmost', True)
    parent.update()
    path = filedialog.asksaveasfilename(
        parent=parent,
        title=title,
        defaultextension=defaultextension,
        initialfile=default_name,
        filetypes=filetypes)
    if need_destroy:
        parent.destroy()
    return path or None

def ask_save_path(title, default_name, defaultextension='', filetypes=None):
    filetypes = filetypes or [('All files', '*')]
    if _provider is not None:
        try:
            return _provider(title=title, default_name=default_name,
                             defaultextension=defaultextension,
                             filetypes=filetypes)
        except Exception:
            pass
    try:
        return _tk_save_path(title, default_name, defaultextension, filetypes)
    except Exception:
        path = input(f'Save path [{default_name}]: ').strip().strip('"')
        return path or default_name
