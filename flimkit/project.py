import json
from dataclasses import dataclass, asdict
from pathlib import Path

#  Leica .sptw sub-folder names to search for tile PTUs 
_SPTW_CANDIDATES = [
    '{stem}.sptw',   # most common: scan-name.sptw
    'PTU.sptw',      # older Leica export style
    'FLIM.sptw',
]

PROJECT_FILENAME = 'project.json'
DEFAULT_OUTPUT_SUBDIR = 'output'


@dataclass
class ScanRecord:
    stem:          str
    scan_type:     str            # "fov" | "xlif"
    source_path:   str            # absolute path to .ptu or .xlif
    ptu_dir:       str | None = None
    out_st:        str | None = None
    output_prefix: str | None = None
    xlsx_path:     str | None = None

    #  derived helpers 

    @property
    def roi_clean(self):
        return self.stem.replace(' ', '_')

    @property
    def session_path(self):
        if self.scan_type == 'fov':
            p = Path(self.source_path)
            candidate = p.parent / f"{p.stem}.roi_session.npz"
        else:  # xlif
            if not self.out_st:
                return None
            candidate = Path(self.out_st) / self.roi_clean / 'roi_session.npz'
        return candidate if candidate.exists() else None

    @property
    def phasor_session_path(self):
        if self.scan_type != 'fov':
            return None
        p = Path(self.source_path)
        candidate = p.parent / f"{p.stem}_phasor.npz"
        return candidate if candidate.exists() else None

    @property
    def has_session(self):
        return self.session_path is not None

    @property
    def has_phasor_session(self):
        return self.phasor_session_path is not None


class ProjectFile:

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir).resolve()
        self.output_base = self.project_dir / DEFAULT_OUTPUT_SUBDIR
        self.scans = {}
        self.config = {}  # per-project config overrides

    # persistence

    @classmethod
    def load_or_create(cls, project_dir):
        pf = cls(project_dir)
        json_path = pf.project_dir / PROJECT_FILENAME
        if json_path.exists():
            try:
                with open(json_path, encoding='utf-8') as fh:
                    data = json.load(fh)
                ob = data.get('output_base')
                if ob:
                    pf.output_base = Path(ob)
                for stem, rec_dict in data.get('scans', {}).items():
                    pf.scans[stem] = ScanRecord(**rec_dict)
                pf.config = data.get('config', {})
            except Exception as exc:
                # Corrupted project.json - start fresh, log the error
                print(f"[Project] Warning: could not read {json_path.name}: {exc}")
        pf._scan_folder()
        return pf

    def save(self):
        self.project_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'output_base': str(self.output_base),
            'scans': {stem: asdict(rec) for stem, rec in self.scans.items()},
        }
        if self.config:
            payload['config'] = self.config
        with open(self.project_dir / PROJECT_FILENAME, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    #  scan discovery

    def _scan_folder(self):
        for ptu in sorted(self.project_dir.glob('*.ptu')):
            if ptu.name.startswith('._'):
                continue
            if ptu.stem not in self.scans:
                # Check for paired .xlsx file (same name, different extension)
                xlsx_file = self.project_dir / f"{ptu.stem}.xlsx"
                xlsx_path = str(xlsx_file) if xlsx_file.exists() and not xlsx_file.name.startswith('._') else None
                
                self.scans[ptu.stem] = ScanRecord(
                    stem          = ptu.stem,
                    scan_type     = 'fov',
                    source_path   = str(ptu),
                    ptu_dir       = None,
                    out_st        = None,
                    output_prefix = None,
                    xlsx_path     = xlsx_path,
                )

        for xlif in sorted(self.project_dir.glob('*.xlif')):
            if xlif.name.startswith('._'):
                continue
            if xlif.stem not in self.scans:
                ptu_dir = self._find_sptw(xlif)
                self.scans[xlif.stem] = ScanRecord(
                    stem          = xlif.stem,
                    scan_type     = 'xlif',
                    source_path   = str(xlif),
                    ptu_dir       = str(ptu_dir) if ptu_dir else None,
                    out_st        = str(self.output_base),
                    output_prefix = None,
                    xlsx_path     = None,
                )

    def _find_sptw(self, xlif):
        base = xlif.parent
        for template in _SPTW_CANDIDATES:
            candidate = base / template.format(stem=xlif.stem)
            if candidate.is_dir():
                return candidate
        return None

    #  post-fit update 

    def update_after_fit(
        self,
        stem,
        *,
        out_st=None,
        output_prefix=None,
        ptu_dir=None,
    ):
        rec = self.scans.get(stem)
        if rec is None:
            return
        if out_st is not None:
            rec.out_st = out_st
        if output_prefix is not None:
            rec.output_prefix = output_prefix
        if ptu_dir is not None:
            rec.ptu_dir = ptu_dir

    def update_after_phasor(self, stem: str):
        pass

    #  convenience

    def default_out_st(self, stem):
        rec = self.scans.get(stem)
        if rec and rec.out_st:
            return rec.out_st
        return str(self.output_base)

    def default_output_prefix(self, stem):
        rec = self.scans.get(stem)
        if rec and rec.output_prefix:
            return rec.output_prefix
        return stem

    def sorted_scans(self):
        yield from sorted(self.scans.items())
