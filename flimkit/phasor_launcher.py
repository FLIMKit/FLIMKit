import os
import sys
from pathlib import Path

import numpy as np

def _yes_no(question):
    import inquirer
    ans = inquirer.prompt([inquirer.List(
        'yesno', message=question, choices=['Yes', 'No'])])
    return ans['yesno'] == 'Yes'

def _ask_path(message, *, optional=False):
    import inquirer
    hint = ' (leave blank to skip)' if optional else ''
    ans = inquirer.prompt([
        inquirer.Path('path',
                      message=f'{message}{hint}',
                      path_type=inquirer.Path.FILE,
                      exists=not optional)])
    val = (ans or {}).get('path', '').strip()
    if not val:
        return None
    return val

def _pick_save_file(title: str, default_name: str) -> str | None:
    try:
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
            defaultextension='.npz',
            initialfile=default_name,
            filetypes=[('NumPy archive', '*.npz'), ('All files', '*')])
        if need_destroy:
            parent.destroy()
        return path or None
    except Exception:
        path = input(f'Save path [{default_name}]: ').strip().strip('"')
        return path or default_name

def save_session(path, *,
                 real_cal,
                 imag_cal,
                 mean,
                 frequency,
                 cursors,
                 params,
                 ptu_file=None,
                 irf_file=None,
                 display_image=None):
    n = len(cursors)
    cursor_g = np.array([c['center_g'] for c in cursors], dtype=float) if n else np.array([], dtype=float)
    cursor_s = np.array([c['center_s'] for c in cursors], dtype=float) if n else np.array([], dtype=float)
    cursor_colors = np.array([c['color'] for c in cursors], dtype='U10') if n else np.array([], dtype='U10')
    save_kw = dict(
        real_cal=real_cal,
        imag_cal=imag_cal,
        mean=mean,
        frequency=np.float64(frequency),
        cursor_g=cursor_g,
        cursor_s=cursor_s,
        cursor_colors=cursor_colors,
        param_radius=np.float64(params.get('radius', 0.05)),
        param_radius_minor=np.float64(params.get('radius_minor', 0.03)),
        param_angle_mode=np.array(params.get('angle_mode', 'semicircle')),
        ptu_file=np.array(ptu_file or ''),
        irf_file=np.array(irf_file or ''),
    )
    if display_image is not None:
        save_kw['display_image'] = np.asarray(display_image)
    np.savez_compressed(path, **save_kw)
    print(f'Session saved → {path}  ({n} cursor(s))')

def load_session(path):
    d = np.load(path, allow_pickle=False)
    cursors = []
    g = d['cursor_g']
    s = d['cursor_s']
    colors = d['cursor_colors']
    for i in range(len(g)):
        cursors.append(dict(
            center_g=float(g[i]),
            center_s=float(s[i]),
            color=str(colors[i]),
        ))
    params = dict(
        radius=float(d['param_radius']),
        radius_minor=float(d['param_radius_minor']),
        angle_mode=str(d['param_angle_mode']),
    )
    return dict(
        real_cal=d['real_cal'],
        imag_cal=d['imag_cal'],
        mean=d['mean'],
        frequency=float(d['frequency']),
        cursors=cursors,
        params=params,
        ptu_file=str(d['ptu_file']) or None,
        irf_file=str(d['irf_file']) or None,
        display_image=d['display_image'] if 'display_image' in d else None,
    )

def get_ptu_active_channels(ptu_path):
    from flimkit.formats import FLIMFile
    ptu = FLIMFile(str(ptu_path), verbose=False)
    records = ptu._load_records()
    special, ch_raw, _, _ = ptu._decode_records(records)
    active_channels = np.unique(ch_raw[~special]).astype(int)
    return sorted(int(channel) for channel in active_channels)

def _prompt_ptu_channel(active_channels):
    import inquirer
    answer = inquirer.prompt([inquirer.List(
        'channel',
        message=(
            f'Multiple channels detected: {active_channels}. '
            'Which one should be used for phasor analysis?'
        ),
        choices=[f'Channel {channel}' for channel in active_channels],
    )])
    if not answer or 'channel' not in answer:
        raise ValueError('No channel selected for phasor analysis')
    return int(str(answer['channel']).split()[-1])

def resolve_ptu_channel(
    ptu_path,
    channel=None,
    *,
    prompt_fn=None,
):
    active_channels = get_ptu_active_channels(ptu_path)
    if not active_channels:
        raise ValueError('No photon channels found in PTU file')
    if channel is not None:
        selected_channel = int(channel)
        if selected_channel not in active_channels:
            raise ValueError(
                f'Channel {selected_channel} is not present in PTU file. '
                f'Available channels: {active_channels}'
            )
        return selected_channel
    if len(active_channels) == 1:
        selected_channel = active_channels[0]
        print(f'Auto-selected channel {selected_channel} (only channel available)')
        return selected_channel
    if prompt_fn is None:
        prompt_fn = _prompt_ptu_channel
    selected_channel = int(prompt_fn(active_channels))
    if selected_channel not in active_channels:
        raise ValueError(
            f'Channel {selected_channel} is not present in PTU file. '
            f'Available channels: {active_channels}'
        )
    return selected_channel

def _process_ptu(ptu_path, irf_path=None, channel=None, phasor_filter=None,
                 filter_kwargs=None):
    from phasorpy.phasor import phasor_from_signal
    from .formats.PTU.tools import signal_from_PTUFile
    from flimkit.formats import FLIMFile
    from .phasor.signal import get_phasor_irf, calibrate_signal_with_irf
    print(f'Loading PTU file: {ptu_path}')
    channel = resolve_ptu_channel(ptu_path, channel=channel)
    print(f'Using channel: {channel}')
    signal = signal_from_PTUFile(ptu_path, dtype=np.uint32, binning=4, channel=channel)
    frequency = float(signal.attrs['frequency'])
    print(f'Computing phasors (frequency = {frequency:.2f} MHz) ...')
    mean, real, imag = phasor_from_signal(signal, axis='H')
    ptu = FLIMFile(str(ptu_path), verbose=False)
    display_image = ptu.raw_pixel_stack(channel=channel, binning=4).sum(axis=-1)
    if irf_path:
        from .phasor.signal import calibrate_signal_with_machine_irf
        irf_path_p = Path(irf_path)
        if irf_path_p.suffix.lower() == '.npy':
            print(f'Calibrating with machine IRF (.npy): {irf_path}')
            real_cal, imag_cal = calibrate_signal_with_machine_irf(
                signal, real, imag, irf_path, frequency)
        else:
            print(f'Calibrating with IRF (.xlsx): {irf_path}')
            irf_time_ns, irf_counts = get_phasor_irf(irf_path)
            real_cal, imag_cal = calibrate_signal_with_irf(
                signal, real, imag, irf_time_ns, irf_counts, frequency)
    else:
        print(' No IRF - using uncalibrated phasor coordinates.')
        real_cal, imag_cal = real, imag
    if phasor_filter:
        from .phasor.filters import phasor_filter as _filter_fn
        print(f'Applying phasor filter: {phasor_filter} ...')
        real_cal, imag_cal = _filter_fn(
            np.asarray(real_cal, dtype=float),
            np.asarray(imag_cal, dtype=float),
            phasor_filter,
            mean=np.asarray(mean, dtype=float),
            **(filter_kwargs or {}),
        )
    return dict(
        real_cal=np.asarray(real_cal),
        imag_cal=np.asarray(imag_cal),
        mean=np.asarray(mean),
        frequency=frequency,
        channel=channel,
        display_image=np.asarray(display_image, dtype=float),
    )

def launch_phasor(ptu_path=None,
                  irf_path=None,
                  machine_irf_path=None,
                  session_path=None,
                  *,
                  channel=None,
                  phasor_filter=None,
                  filter_kwargs=None,
                  min_photons=0.01,
                  max_cursors=6,
                  figsize=(8, 5)):
    from .phasor.interactive import phasor_cursor_tool
    from flimkit.formats import file_modality
    initial_cursors = None
    initial_params = None
    src_ptu = ptu_path
    src_irf = irf_path
    if session_path is None and ptu_path is None:
        import inquirer
        mode = inquirer.prompt([inquirer.List(
            'mode',
            message='What would you like to do?',
            choices=[
                'Analyse a new PTU file',
                'Resume a saved session (.npz)',
            ],
        )])['mode']
        if mode.startswith('Resume'):
            session_path = _ask_path('Path to saved .npz session')
            if session_path is None:
                print('No file specified - aborting.')
                return {}
        else:
            ptu_path = _ask_path('Path to PTU file')
            if ptu_path is None:
                print('No file specified - aborting.')
                return {}
    if session_path:
        print(f'Loading session: {session_path}')
        sess = load_session(session_path)
        data = dict(
            real_cal=sess['real_cal'],
            imag_cal=sess['imag_cal'],
            mean=sess['mean'],
            frequency=sess['frequency'],
            display_image=sess.get('display_image'),
        )
        initial_cursors = sess['cursors'] or None
        initial_params = sess['params']
        src_ptu = sess.get('ptu_file')
        src_irf = sess.get('irf_file')
        print(f'  frequency = {data['frequency']:.2f} MHz, '
              f'{len(sess['cursors'])} cursor(s) restored')
    elif ptu_path and file_modality(ptu_path) == 'frequency':
        from .phasor.signal import process_ifli
        data = process_ifli(ptu_path, phasor_filter=phasor_filter,
                            filter_kwargs=filter_kwargs, channel=channel)
    else:
        if irf_path is None and machine_irf_path is None:
            choices = [
                'XLSX IRF (analytical model)',
                'Machine IRF (.npy pre-built)',
                'No IRF (uncalibrated)',
            ]
            irf_choice = inquirer.prompt([inquirer.List(
                'irf', message='IRF calibration source?', choices=choices
            )])['irf']
            if irf_choice.startswith('XLSX'):
                irf_path = _ask_path('Path to IRF Excel file (.xlsx)')
            elif irf_choice.startswith('Machine'):
                machine_irf_path = _ask_path('Path to machine IRF (.npy)')
        effective_irf = irf_path or machine_irf_path
        src_irf = effective_irf
        effective_irf = irf_path or machine_irf_path
        data = _process_ptu(ptu_path, effective_irf, channel=channel,
                            phasor_filter=phasor_filter,
                            filter_kwargs=filter_kwargs)
    _data = data

    def _save_callback(state, params):
        stem = Path(src_ptu).stem if src_ptu else 'phasor_session'
        default_name = f'{stem}_session.npz'
        out = _pick_save_file('Save phasor session', default_name)
        if out:
            save_session(
                out,
                real_cal=_data['real_cal'],
                imag_cal=_data['imag_cal'],
                mean=_data['mean'],
                frequency=_data['frequency'],
                cursors=state['cursors'],
                params=params,
                ptu_file=src_ptu,
                irf_file=src_irf,
                display_image=_data.get('display_image'),
            )
    state = phasor_cursor_tool(
        data['real_cal'],
        data['imag_cal'],
        data['mean'],
        data['frequency'],        display_image=data.get('display_image'),        min_photons=min_photons,
        max_cursors=max_cursors,
        figsize=figsize,
        initial_cursors=initial_cursors,
        initial_params=initial_params,
        on_save=_save_callback,
    )
    return state

def phasor_inquire():
    print('\n Interactive Phasor Analysis')
    return launch_phasor()

if __name__ == '__main__':
    phasor_inquire()
