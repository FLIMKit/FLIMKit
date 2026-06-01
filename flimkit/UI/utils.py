from __future__ import annotations
import re
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path
from typing import Optional
import numpy as np
try:
    from tkinterdnd2 import DND_FILES, DND_TEXT
    HAS_DND = True
except ImportError:
    HAS_DND = False


_cfg: dict = {}


def _C() -> dict:
    if not _cfg:
        from flimkit.configs import (
            n_exp, Tau_min, Tau_max, D_mode, binning_factor,
            MIN_PHOTONS_PERPIX, Optimizer, lm_restarts, de_population,
            de_maxiter, n_workers, OUT_NAME, IRF_BINS, IRF_FIT_WIDTH,
            IRF_FWHM, channels, TAU_DISPLAY_MIN, TAU_DISPLAY_MAX,
            INTENSITY_DISPLAY_MIN, INTENSITY_DISPLAY_MAX,
            MACHINE_IRF_DIR, MACHINE_IRF_DEFAULT_PATH,
            MACHINE_IRF_ALIGN_ANCHOR, MACHINE_IRF_REDUCER,
        )
        _cfg.update(
            n_exp=n_exp, Tau_min=Tau_min, Tau_max=Tau_max, D_mode=D_mode,
            binning_factor=binning_factor, MIN_PHOTONS_PERPIX=MIN_PHOTONS_PERPIX,
            Optimizer=Optimizer, lm_restarts=lm_restarts,
            de_population=de_population, de_maxiter=de_maxiter,
            n_workers=n_workers, OUT_NAME=OUT_NAME,
            IRF_BINS=IRF_BINS, IRF_FIT_WIDTH=IRF_FIT_WIDTH, IRF_FWHM=IRF_FWHM,
            channels=channels,
            TAU_DISPLAY_MIN=TAU_DISPLAY_MIN, TAU_DISPLAY_MAX=TAU_DISPLAY_MAX,
            INTENSITY_DISPLAY_MIN=INTENSITY_DISPLAY_MIN,
            INTENSITY_DISPLAY_MAX=INTENSITY_DISPLAY_MAX,
            MACHINE_IRF_DIR=MACHINE_IRF_DIR,
            MACHINE_IRF_DEFAULT_PATH=MACHINE_IRF_DEFAULT_PATH,
            MACHINE_IRF_ALIGN_ANCHOR=MACHINE_IRF_ALIGN_ANCHOR,
            MACHINE_IRF_REDUCER=MACHINE_IRF_REDUCER,
        )
    return _cfg



def _reconstruct_dict_from_session(session_data: dict, key: str) -> dict:
    """
    Inverse of the hoisting done in _save_roi_progress.
    Reconstructs a dict from JSON + hoisted numpy arrays stored separately.
    
    Args:
        session_data: The session/fit result dict containing "key_json" and "key_arr_*" entries
        key: Base key name (e.g. "global_summary" or "pixel_maps")
    
    Returns:
        Reconstructed dict with arrays reattached
    """
    import json
    result = {}
    json_str = session_data.get(f"{key}_json")
    if json_str:
        if isinstance(json_str, (bytes, np.ndarray)):
            json_str = json_str.item() if hasattr(json_str, 'item') else json_str.decode()
        try:
            result = json.loads(json_str)
        except:
            pass
    
    prefix = f"{key}_arr_"
    for skey, sval in session_data.items():
        if skey.startswith(prefix) and isinstance(sval, np.ndarray):
            result[skey[len(prefix):]] = sval
    
    return result


def _safe_array_from_json(value) -> np.ndarray:
    """
    Safely convert a value that may be a string representation of an array
    back to a real numpy array. Handles numpy scalar wrappers.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, np.ndarray)):
        if hasattr(value, 'item'):
            value = value.item()
        else:
            value = value.decode() if isinstance(value, bytes) else str(value)
    if isinstance(value, str):
        try:
            import re
            # Try to parse numpy array string format: [1.0 2.0 3.0] with optional formatting
            value = re.sub(r'\s+', ' ', value.strip())
            value = value.replace('e+', 'e+').replace('e-', 'e-')
            return np.fromstring(value.strip('[]'), sep=' ')
        except:
            pass
    return np.asarray(value)


def _parse_summary(captured_log: str) -> list:
    """
    Parse the captured stdout/stderr for a summary table.
    This is a placeholder – replace with actual parsing if needed.
    Returns a list of (parameter, value, unit) rows.
    """
    rows = []
    # Example: find lines like "tau1 = 2.45 ns"
    for line in captured_log.splitlines():
        if "tau" in line.lower() and "=" in line:
            parts = line.split("=", 1)
            if len(parts) == 2:
                param = parts[0].strip()
                rest = parts[1].strip()
                val_unit = rest.split()
                if len(val_unit) >= 2:
                    rows.append((param, val_unit[0], val_unit[1]))
                else:
                    rows.append((param, rest, ""))
    return rows


class _Redirect:
    """Redirect stdout/stderr to ScrolledText; batches updates for performance (thread-safe)."""

    def __init__(self, widget: scrolledtext.ScrolledText, buf: list, root=None, is_stderr=False):
        self.widget = widget
        self.buf    = buf
        self.root   = root  # For thread-safe GUI updates
        self._is_stderr = is_stderr
        self._batch = []  # Accumulate text before writing
        self._batch_size = 5000  # characters, or time-based flush
        self._last_flush = time.time()
        self._flush_interval = 0.5  # seconds

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        self._batch.append(text)

        # Forward stderr content to crash handler log
        if self._is_stderr:
            try:
                from flimkit.utils.crash_handler import log_event
                log_event(f"STDERR: {text.rstrip()}", level="warning")
            except Exception:
                pass
        
        # Flush if batch is large or timeout elapsed
        should_flush = False
        if len("".join(self._batch)) >= self._batch_size:
            should_flush = True
        elif time.time() - self._last_flush >= self._flush_interval:
            should_flush = True
        
        if should_flush:
            self._flush_batch()

    def _flush_batch(self):
        if not self._batch:
            return
        text = "".join(self._batch)
        self._batch.clear()
        
        # Use root.after() for thread-safe GUI updates if root is available
        if self.root:
            self.root.after(0, self._update_widget, text)
        else:
            # Fallback to direct update (not thread-safe but works in single-threaded context)
            self._update_widget(text)
    
    def _update_widget(self, text):
        """Update widget from main thread."""
        try:
            self.widget.configure(state="normal")
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.configure(state="disabled")
            self.widget.update_idletasks()
        except Exception:
            pass  # Widget may have been destroyed
        self._last_flush = time.time()

    def flush(self):
        self._flush_batch()


class _FileRedirect:
    """Redirect stdout/stderr to a file for performance (no widget updates)."""

    def __init__(self, filepath: str, buf: list):
        self.filepath = filepath
        self.buf = buf
        self._file = None
        try:
            self._file = open(filepath, 'w', buffering=1)  # Line buffering
        except Exception:
            pass

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        if self._file:
            try:
                self._file.write(text)
            except Exception:
                pass

    def flush(self):
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass

    def close(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


class _FileTailer:
    """Stream log file content to a Text widget in real-time."""

    def __init__(self, filepath: str, widget: scrolledtext.ScrolledText, update_interval_ms: int = 200):
        self.filepath = filepath
        self.widget = widget
        self.update_interval_ms = update_interval_ms
        self._file = None
        self._last_pos = 0
        self._running = False

    def start(self, root):
        """Start tailing the file and updating the widget."""
        self._running = True
        self._poll_file(root)

    def _poll_file(self, root):
        """Poll the file for new content and update widget."""
        if not self._running:
            return
        
        try:
            if not Path(self.filepath).exists():
                root.after(self.update_interval_ms, lambda: self._poll_file(root))
                return
            
            # Read new content from file
            with open(self.filepath, 'r') as f:
                f.seek(self._last_pos)
                new_content = f.read()
                self._last_pos = f.tell()
            
            # Update widget if there's new content
            if new_content:
                self.widget.configure(state="normal")
                self.widget.insert(tk.END, new_content)
                self.widget.see(tk.END)
                self.widget.configure(state="disabled")
                self.widget.update_idletasks()
        except Exception:
            pass
        
        # Schedule next poll
        root.after(self.update_interval_ms, lambda: self._poll_file(root))

    def stop(self):
        """Stop tailing the file."""
        self._running = False


PAD = dict(padx=8, pady=4)


def _browse_file(var, title="Select file", filetypes=(("All", "*.*"),)):
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if p:
        var.set(p)


def _browse_dir(var, title="Select directory"):
    p = filedialog.askdirectory(title=title)
    if p:
        var.set(p)


def _row(parent, label, var, row, browse_fn, width=45, state="normal"):
    ttk.Label(parent, text=label).grid(
        row=row, column=0, sticky="e", padx=6, pady=3)
    e = ttk.Entry(parent, textvariable=var, width=width, state=state)
    e.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
    ttk.Button(parent, text="Browse...", command=browse_fn).grid(
        row=row, column=2, padx=4, pady=3)
    
    # Add drag-and-drop support if available
    if HAS_DND:
        try:
            def _drop_handler(event):
                data = event.data.strip()
                if data.startswith("{") and data.endswith("}"):
                    data = data[1:-1]
                var.set(data)
            
            e.drop_target_register(DND_FILES, DND_TEXT)
            e.dnd_bind("<<Drop>>", _drop_handler)
        except Exception:
            pass
    
    return e


def _section(parent, text: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(parent, text=f"  {text}  ", padding=(10, 6))


def _tog(bvar: tk.BooleanVar, entry: ttk.Entry):
    entry.configure(state="normal" if bvar.get() else "disabled")


def _flt(sv: tk.StringVar) -> Optional[float]:
    v = sv.get().strip()
    return float(v) if v and v.lower() != "none" else None


def _thresh(bvar: tk.BooleanVar, sv: tk.StringVar):
    """Return threshold value, or None."""
    if not bvar.get():
        return None
    v = sv.get().strip()
    return int(v) if v else None
