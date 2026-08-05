from pathlib import Path

import pandas as pd


def _csv_layout(path: str | Path) -> tuple[str, int]:
    with Path(path).open(encoding='utf-8-sig') as export:
        for line in export:
            if 'time [' not in line.lower():
                continue
            counts = {delimiter: line.count(delimiter) for delimiter in (',', ';')}
            delimiter = max(counts, key=lambda item: counts[item])
            if counts[delimiter]:
                return delimiter, counts[delimiter] + 1
    raise ValueError(
        f'Could not parse {Path(path).name}: no delimited LAS X header '
        'containing "Time [" was found.'
    )


def _read_export(path: str | Path, header=None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == '.xlsx':
        return pd.read_excel(path, sheet_name=0, header=header)
    if suffix == '.csv':
        delimiter, n_columns = _csv_layout(path)
        decimal = ',' if delimiter == ';' else '.'
        if header is None:
            return pd.read_csv(path, sep=delimiter, decimal=decimal, header=None,
                               names=range(n_columns), encoding='utf-8-sig')
        return pd.read_csv(path, sep=delimiter, decimal=decimal, header=header,
                           encoding='utf-8-sig')
    raise ValueError(
        f'Unsupported LAS X export format: {suffix or "<none>"}. '
        'Expected .xlsx or .csv.'
    )


def load_irf_export(path: str | Path, debug: bool = False) -> dict:
    df_raw = _read_export(path, header=None)

    if debug:
        print(f"    Raw export shape: {df_raw.shape}")
        print(f"    First 5 rows:")
        for i in range(min(5, len(df_raw))):
            vals = [str(v) for v in df_raw.iloc[i].values if pd.notna(v)]
            print(f"      row {i}: {vals}")

    # Find header row: look for a cell that is exactly (or starts with) "Time [ns]"
    # Must NOT match "Lifetime" - require the word starts with "time ["
    header_row = None
    for i, row in df_raw.iterrows():
        vals = [str(v).strip().lower() for v in row if pd.notna(v)]
        if any(v.startswith('time [') for v in vals):
            header_row = i
            break

    if header_row is None:
        print(f"    No row starting with 'Time [' found - trying row 0 as fallback")
        header_row = 0

    if debug:
        print(f"    Detected header row: {header_row}")

    df        = _read_export(path, header=header_row)
    df        = df.dropna(axis=1, how='all')
    col_names = list(df.columns)

    if debug:
        print(f"    Columns after read: {col_names}")

    time_cols  = [c for c in col_names if str(c).lower().startswith('time [')]
    decay_cols = [c for c in col_names if 'decay'    in str(c).lower() and
                                          'counts'   in str(c).lower()]
    irf_cols   = [c for c in col_names if 'irf'      in str(c).lower() and
                                          'counts'   in str(c).lower()]
    fit_cols   = [c for c in col_names if 'fit'      in str(c).lower() and
                                          'counts'   in str(c).lower()]
    res_cols   = [c for c in col_names if 'resid'    in str(c).lower() and
                                          'counts'   in str(c).lower()]

    if debug:
        print(f"    time_cols : {time_cols}")
        print(f"    decay_cols: {decay_cols}")
        print(f"    irf_cols  : {irf_cols}")
        print(f"    fit_cols  : {fit_cols}")
        print(f"    res_cols  : {res_cols}")

    def _safe(col):
        if col is None:
            return None
        arr = df[col].dropna().values
        try:
            return arr.astype(float)
        except (ValueError, TypeError):
            return None

    out = {
        'decay_t': _safe(time_cols[0]  if len(time_cols) > 0 else None),
        'decay_c': _safe(decay_cols[0] if len(decay_cols) > 0 else None),
        'irf_t':   _safe(time_cols[1]  if len(time_cols) > 1 else None),
        'irf_c':   _safe(irf_cols[0]   if len(irf_cols)  > 0 else None),
        'fit_t':   _safe(time_cols[2]  if len(time_cols) > 2 else None),
        'fit_c':   _safe(fit_cols[0]   if len(fit_cols)  > 0 else None),
        'res_t':   _safe(time_cols[3]  if len(time_cols) > 3 else None),
        'res_c':   _safe(res_cols[0]   if len(res_cols)  > 0 else None),
    }

    for k, v in out.items():
        status = f"{len(v)} pts" if v is not None else 'absent'
        print(f"    {k:12s}: {status}")

    return out


def load_xlsx(path: str | Path, debug: bool = False) -> dict:
    """Backward-compatible wrapper for LAS X Excel exports."""
    return load_irf_export(path, debug=debug)
