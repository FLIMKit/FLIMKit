class AppState:
    def values(self) -> dict:
        # Snapshot of every tk variable currently held, name -> current value.
        out = {}
        for nam, var in self.__dict__.items():
            try:
                out[nam] = var.get()
            except Exception:
                pass
        return out
