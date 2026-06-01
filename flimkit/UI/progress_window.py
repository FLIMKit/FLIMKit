import tkinter as tk
from tkinter import ttk
import threading

class ProgressWindow:
    def __init__(self, parent, task_name="Working..."):
        self.parent = parent
        self.cancelled = threading.Event()
        self.top = tk.Toplevel(parent)
        self.top.title(task_name)
        self.top.geometry("400x120")
        self.top.resizable(True, True)
        self.top.transient(parent)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)

        # Make the window scalable with columns and rows
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(1, weight=1)

        ttk.Label(self.top, text=task_name, font=("Arial", 12, "bold")).grid(
            row=0, column=0, pady=(16, 8), sticky="ew", padx=16)
        self.progress = ttk.Progressbar(self.top, orient="horizontal", mode="determinate")
        self.progress.grid(row=1, column=0, pady=6, padx=16, sticky="ew")
        self.status = tk.StringVar(value="Starting...")
        ttk.Label(self.top, textvariable=self.status).grid(
            row=2, column=0, pady=(2, 8), padx=16, sticky="ew")
        self.btn_cancel = ttk.Button(self.top, text="Cancel", command=self.cancel)
        self.btn_cancel.grid(row=3, column=0, pady=(0, 8), sticky="ew", padx=16)

    def set_progress(self, value, maximum=None):
        if maximum is not None:
            self.progress["maximum"] = maximum
        self.progress["value"] = value
        self.status.set(f"{int(value)}/{int(self.progress['maximum'])} completed")
        self.top.update_idletasks()

    def set_status(self, msg):
        self.status.set(msg)
        self.top.update_idletasks()

    def cancel(self):
        self.cancelled.set()
        self.set_status("Cancelling...")
        self.btn_cancel.config(state="disabled")

    def close(self):
        self.top.grab_release()
        self.top.destroy()

    def was_cancelled(self):
        return self.cancelled.is_set()


class ProgressWindowManager:
    """Manages nested progress windows for sub-operations, callable from worker threads."""
    
    def __init__(self, root):
        self.root = root
        self.windows = {}  # task_id -> ProgressWindow
        self.counter = 0
        self.lock = threading.Lock()
    
    def create_progress_window(self, task_name="Processing..."):
        """Create a progress window and return its ID. Thread-safe."""
        with self.lock:
            window_id = self.counter
            self.counter += 1
        
        # Create window on main thread
        window_ref = [None]
        event = threading.Event()
        
        def create_window():
            window_ref[0] = ProgressWindow(self.root, task_name=task_name)
            with self.lock:
                self.windows[window_id] = window_ref[0]
            event.set()
        
        self.root.after(0, create_window)
        event.wait(timeout=5)  # Wait up to 5 seconds for window to be created
        return window_id
    
    def update_progress(self, window_id, current, total):
        """Update progress for a window. Thread-safe."""
        def update():
            if window_id in self.windows:
                try:
                    self.windows[window_id].set_progress(current, maximum=total)
                except Exception:
                    pass  # Window may have been closed
        
        self.root.after(0, update)
    
    def set_status(self, window_id, msg):
        """Set status message for a window. Thread-safe."""
        def update():
            if window_id in self.windows:
                try:
                    self.windows[window_id].set_status(msg)
                except Exception:
                    pass
        
        self.root.after(0, update)
    
    def close_window(self, window_id):
        """Close a progress window. Thread-safe."""
        def close():
            if window_id in self.windows:
                try:
                    self.windows[window_id].close()
                except Exception:
                    pass
                finally:
                    with self.lock:
                        self.windows.pop(window_id, None)
        
        self.root.after(0, close)
    
    def close_all(self):
        """Close all progress windows."""
        with self.lock:
            ids = list(self.windows.keys())
        for wid in ids:
            self.close_window(wid)
