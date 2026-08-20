import json
import re
import numpy as np

def _reconstruct_dict_from_session(session_data: dict, key: str) -> dict:
    result = {}
    json_str = session_data.get(f'{key}_json')
    if json_str:
        if isinstance(json_str, (bytes, np.ndarray)):
            json_str = json_str.item() if hasattr(json_str, 'item') else json_str.decode()
        try:
            result = json.loads(json_str)
        except Exception:
            pass
    prefix = f'{key}_arr_'
    for skey, sval in session_data.items():
        if skey.startswith(prefix) and isinstance(sval, np.ndarray):
            result[skey[len(prefix):]] = sval
    return result

def _safe_array_from_json(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, np.ndarray)):
        if hasattr(value, 'item'):
            value = value.item()
        else:
            value = value.decode() if isinstance(value, bytes) else str(value)
    if isinstance(value, str):
        try:
            value = re.sub(r'\s+', ' ', value.strip())
            value = value.replace('e+', 'e+').replace('e-', 'e-')
            return np.fromstring(value.strip('[]'), sep=' ')
        except Exception:
            pass
    return np.asarray(value)

def _parse_summary(captured_log: str) -> list:
    rows = []
    for line in captured_log.splitlines():
        if 'tau' in line.lower() and '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                param = parts[0].strip()
                rest = parts[1].strip()
                val_unit = rest.split()
                if len(val_unit) >= 2:
                    rows.append((param, val_unit[0], val_unit[1]))
                else:
                    rows.append((param, rest, ''))
    return rows
