import re
import numpy as np
from pathlib import Path

TILE_ONLY_PATTERN = re.compile(r'^(?P<base>.+?)_s(?P<s>\d+)\.ptu$', re.IGNORECASE)

def parse_series_name(name):
    from ...utils.batch_fit import parse_timelapse_filename
    name = str(name)
    parsed = parse_timelapse_filename(name)
    if parsed is not None:
        region, t, s, z = parsed
        return {
            'file': name,
            'base': region,
            't': t,
            's': s or 1,
            'z': z or 1,
            'has_t': True,
            'has_z': z > 0,
        }
    m = TILE_ONLY_PATTERN.match(name)
    if m is None:
        return None
    return {
        'file': name,
        'base': m.group('base'),
        't': 1,
        's': int(m.group('s')),
        'z': 1,
        'has_t': False,
        'has_z': False,
    }

def index_ptu_series(ptu_dir, ptu_basename=None):
    ptu_dir = Path(ptu_dir)
    entries = []
    for path in sorted(ptu_dir.glob('*.ptu')):
        parsed = parse_series_name(path.name)
        if parsed is None:
            continue
        if ptu_basename is not None and parsed['base'] != ptu_basename:
            continue
        entries.append(parsed)
    if not entries:
        raise RuntimeError(
            f'No PTU files matching <base>_s<N>[_z<N>] found in {ptu_dir}'
            + (f' for basename {ptu_basename!r}' if ptu_basename else ''))
    bases = sorted({e['base'] for e in entries})
    if len(bases) > 1:
        raise RuntimeError(
            f'Multiple series in {ptu_dir}: {bases}. Pass ptu_basename to pick one.')
    planes = {}
    for e in entries:
        planes.setdefault((e['t'], e['z']), []).append(e)
    for key in planes:
        planes[key].sort(key=lambda e: e['s'])
    tile_counts = {len(v) for v in planes.values()}
    tiles = sorted({e['s'] for e in entries})
    if len(tiles) < 2:
        raise RuntimeError(
            f'{bases[0]!r} has a single position, so there is nothing to stitch. '
            'Use the timelapse or z-stack batch fit for single-position stacks.')
    return {
        'base': bases[0],
        'planes': planes,
        'timepoints': sorted({t for t, _ in planes}),
        'z_planes': sorted({z for _, z in planes}),
        'tiles': tiles,
        'n_files': len(entries),
        'is_ragged': len(tile_counts) > 1,
        'has_t': any(e['has_t'] for e in entries),
        'has_z': any(e['has_z'] for e in entries),
    }

def describe_series(index):
    n_t = len(index['timepoints'])
    n_z = len(index['z_planes'])
    n_s = len(index['tiles'])
    return (f"{index['base']!r}: {n_t} timepoint(s) x {n_s} tile(s) x {n_z} z-plane(s) "
            f"= {index['n_files']} files")

def _overlap_score(a, b, dy, dx, min_px):
    h, w = a.shape
    if abs(dy) >= h or abs(dx) >= w:
        return None
    pa = a[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
    pb = b[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    if pa.size < min_px:
        return None
    pa = pa - pa.mean()
    pb = pb - pb.mean()
    den = np.sqrt(float((pa * pa).sum()) * float((pb * pb).sum()))
    if den <= 0:
        return None
    return float((pa * pb).sum() / den), pa.size

def register_tile_pair(image_a, image_b, coarse_step=4, refine_radius=8,
                       min_overlap_frac=0.05, n_candidates=8):
    a = np.log1p(np.asarray(image_a, dtype=float))
    b = np.log1p(np.asarray(image_b, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f'Tile shapes differ: {a.shape} vs {b.shape}')
    h, w = a.shape
    min_px = max(1, int(min_overlap_frac * a.size))
    coarse = []
    for dy in range(-h + 1, h, coarse_step):
        for dx in range(-w + 1, w, coarse_step):
            scored = _overlap_score(a, b, dy, dx, min_px)
            if scored is None:
                continue
            coarse.append((scored[0], dy, dx, scored[1]))
    if not coarse:
        raise RuntimeError(
            f'No candidate shift kept at least {100 * min_overlap_frac:.0f}% overlap')
    coarse.sort(key=lambda c: (-c[0], -c[3]))
    best = None
    for _, cy, cx, _ in coarse[:n_candidates]:
        for dy in range(cy - refine_radius, cy + refine_radius + 1):
            for dx in range(cx - refine_radius, cx + refine_radius + 1):
                scored = _overlap_score(a, b, dy, dx, min_px)
                if scored is None:
                    continue
                r, n = scored
                if best is None or r > best[0] + 1e-9 or (
                        abs(r - best[0]) <= 1e-9 and n > best[3]):
                    best = (r, dy, dx, n)
    r, dy, dx, n_px = best
    return {
        'dy': int(dy),
        'dx': int(dx),
        'correlation': float(r),
        'overlap_px': int(n_px),
        'overlap_frac': float(n_px) / float(a.size),
    }

def recover_tile_positions(tile_images, min_correlation=0.3):
    if len(tile_images) < 2:
        raise ValueError('Need at least two tiles to recover positions')
    shifts = {0: (0, 0)}
    pairs = []
    for i in range(1, len(tile_images)):
        reg = register_tile_pair(tile_images[i - 1], tile_images[i])
        if reg['correlation'] < min_correlation:
            raise RuntimeError(
                f'Tile {i-1} to {i} registration too weak '
                f"(r={reg['correlation']:.3f} < {min_correlation}); "
                f'supply tile positions from a .lif or .xlif instead')
        prev_y, prev_x = shifts[i - 1]
        shifts[i] = (prev_y + reg['dy'], prev_x + reg['dx'])
        pairs.append(reg)
    ys = [shifts[i][0] for i in range(len(tile_images))]
    xs = [shifts[i][1] for i in range(len(tile_images))]
    min_y, min_x = min(ys), min(xs)
    positions = [{'pixel_y': ys[i] - min_y, 'pixel_x': xs[i] - min_x}
                 for i in range(len(tile_images))]
    return positions, pairs

def refine_tile_positions(tile_images, positions, radius=60,
                          min_overlap_frac=0.05):
    if len(tile_images) < 2:
        return [dict(p) for p in positions], []
    imgs = [np.log1p(np.asarray(im, dtype=float)) for im in tile_images]
    min_px = max(1, int(min_overlap_frac * imgs[0].size))
    out = [dict(positions[0])]
    refinements = []
    for i in range(1, len(imgs)):
        prev, cur = positions[i - 1], positions[i]
        dy0 = cur['pixel_y'] - prev['pixel_y']
        dx0 = cur['pixel_x'] - prev['pixel_x']
        best = None
        for dy in range(dy0 - radius, dy0 + radius + 1):
            for dx in range(dx0 - radius, dx0 + radius + 1):
                scored = _overlap_score(imgs[i - 1], imgs[i], dy, dx, min_px)
                if scored is None:
                    continue
                r, n = scored
                if best is None or r > best[0]:
                    best = (r, dy, dx, n)
        if best is None:
            out.append(dict(cur))
            refinements.append(None)
            continue
        r, dy, dx, n = best
        start = _overlap_score(imgs[i - 1], imgs[i], dy0, dx0, min_px)
        out.append({**cur,
                    'pixel_y': out[i - 1]['pixel_y'] + dy,
                    'pixel_x': out[i - 1]['pixel_x'] + dx})
        refinements.append({
            'shift_y': dy - dy0,
            'shift_x': dx - dx0,
            'correlation': float(r),
            'correlation_before': float(start[0]) if start else float('nan'),
            'overlap_px': int(n),
        })
    min_y = min(p['pixel_y'] for p in out)
    min_x = min(p['pixel_x'] for p in out)
    for p in out:
        p['pixel_y'] -= min_y
        p['pixel_x'] -= min_x
    return out, refinements

def _tile_intensity(ptu_path, rotate_cw, binning):
    from .reader import PTUFile
    ptu = PTUFile(str(ptu_path), verbose=False)
    ptu.summed_decay()
    img = ptu.intensity_image(channel=ptu.photon_channel, binning=binning)
    if rotate_cw:
        img = np.rot90(img, k=-1)
    return img

def recover_series_positions(ptu_dir, index, plane=None, rotate_tiles=True,
                             binning=1, min_correlation=0.3, verbose=True):
    ptu_dir = Path(ptu_dir)
    if plane is None:
        mid_t = index['timepoints'][len(index['timepoints']) // 2]
        mid_z = index['z_planes'][len(index['z_planes']) // 2]
        plane = (mid_t, mid_z)
    entries = index['planes'].get(plane)
    if not entries:
        raise RuntimeError(f'Plane {plane} not present in series index')
    if verbose:
        print(f'Recovering tile positions from t={plane[0]} z={plane[1]} '
              f'({len(entries)} tiles)...')
    images = [_tile_intensity(ptu_dir / e['file'], rotate_tiles, binning)
              for e in entries]
    positions, pairs = recover_tile_positions(images, min_correlation=min_correlation)
    tile_positions = []
    for e, pos in zip(entries, positions):
        tile_positions.append({
            'file': e['file'],
            's': e['s'],
            'scan_index': e['s'] - 1,
            'field_x': 0,
            'field_y': e['s'] - 1,
            'pos_x': 0.0,
            'pos_y': 0.0,
            'pixel_x': pos['pixel_x'],
            'pixel_y': pos['pixel_y'],
        })
    if verbose:
        for i, reg in enumerate(pairs):
            print(f"  s{entries[i]['s']} to s{entries[i+1]['s']}: "
                  f"dy={reg['dy']} dx={reg['dx']} r={reg['correlation']:.3f} "
                  f"overlap={100 * reg['overlap_frac']:.1f}%")
    return tile_positions, pairs

def plane_tile_positions(reference_positions, entries):
    by_tile = {p['s']: p for p in reference_positions}
    out = []
    for e in sorted(entries, key=lambda x: x['s']):
        ref = by_tile.get(e['s'])
        if ref is None:
            raise RuntimeError(
                f"Tile s{e['s']} has no recovered position; the reference plane "
                f'was missing that tile')
        out.append({**ref, 'file': e['file']})
    return out
