import threading
import traceback
from pathlib import Path
import numpy as np
import panel as pn
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from flimkit.UI.utils import _C
from flimkit.web.args import build_fov_args

pn.extension('tabulator', notifications=True)

BG = 'black'
FG = 'white'

def style_dark(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(FG)

def style_cbar(cbar):
    cbar.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
    cbar.ax.yaxis.label.set_color(FG)
    cbar.outline.set_edgecolor(FG)

def blank_figure(msg):
    fig = Figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.text(0.5, 0.5, msg, ha='center', va='center', color='#aaa', fontsize=9)
    ax.set_axis_off()
    return fig

def intensity_figure(ptu_path):
    from flimkit.image.tools import make_intensity_image
    img = make_intensity_image(ptu_path, rotate_90_cw=False, save_image=False)
    fig = Figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap='gray')
    ax.set_title(f'{Path(ptu_path).name}  {img.shape[1]}x{img.shape[0]}', fontsize=8)
    ax.set_axis_off()
    style_dark(fig, ax)
    style_cbar(fig.colorbar(im, ax=ax, fraction=0.046, label='photons'))
    fig.tight_layout()
    return fig

def decay_figure(res):
    t = res.get('time_ns')
    d = res.get('decay')
    if t is None or d is None:
        return blank_figure('no decay')
    fig = Figure(figsize=(4.5, 3.2))
    ax = fig.add_subplot(111)
    ax.semilogy(t, np.clip(d, 1e-1, None), lw=0.8, color='#4fc3f7')
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('photons')
    ax.set_title('summed decay', fontsize=9)
    ax.grid(True, which='both', color='#333', lw=0.4)
    style_dark(fig, ax)
    fig.tight_layout()
    return fig

def lifetime_map_key(maps):
    key = next((k for k in ('tau_mean_int', 'tau_mean_amp') if k in maps), None)
    if key is None:
        key = next((k for k in maps if isinstance(maps[k], np.ndarray) and np.asarray(maps[k]).ndim == 2), None)
    return key

def autoscale(res, key):
    maps = res.get('pixel_maps')
    if not maps:
        return None, None
    key = key or lifetime_map_key(maps)
    if key is None or key not in maps:
        return None, None
    arr = np.asarray(maps[key], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)

def lifetime_figure(res, vmin=None, vmax=None, key=None):
    maps = res.get('pixel_maps')
    if not maps:
        return blank_figure('per-pixel not run\n(mode = summed or perPixel)')
    key = key or lifetime_map_key(maps)
    if key is None or key not in maps:
        return blank_figure('no 2-D lifetime map')
    cfg = _C()
    arr = np.asarray(maps[key], dtype=float)
    if vmin is None:
        vmin = cfg['TAU_DISPLAY_MIN']
    if vmax is None:
        vmax = cfg['TAU_DISPLAY_MAX']
    fig = Figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(key, fontsize=9)
    ax.set_axis_off()
    style_dark(fig, ax)
    style_cbar(fig.colorbar(im, ax=ax, fraction=0.046, label='ns'))
    fig.tight_layout()
    return fig

def _seq(g, key):
    val = g.get(key)
    if val is None:
        return []
    return list(np.atleast_1d(val))

def summary_rows(res):
    g = res.get('global_summary') or {}
    rows = []
    for i, tau in enumerate(_seq(g, 'taus_ns')):
        rows.append({'quantity': f'tau{i+1}', 'value': f'{float(tau):.3f}', 'unit': 'ns'})
    for i, amp in enumerate(_seq(g, 'amps')):
        rows.append({'quantity': f'A{i+1}', 'value': f'{float(amp):.3f}', 'unit': ''})
    for i, frac in enumerate(_seq(g, 'fractions')):
        rows.append({'quantity': f'f{i+1}', 'value': f'{float(frac):.3f}', 'unit': ''})
    for key, label, unit in (('tau_mean_amp_ns', 'tau mean (amp)', 'ns'),
                             ('tau_mean_int_ns', 'tau mean (int)', 'ns'),
                             ('reduced_chi2', 'reduced chi2', ''),
                             ('reduced_chi2_pearson', 'reduced chi2 (Pearson)', ''),
                             ('t0_ns', 't0', 'ns'),
                             ('irf_fwhm_eff_ns', 'IRF FWHM (eff.)', 'ns')):
        val = g.get(key)
        if isinstance(val, (int, float)) and val == val:
            rows.append({'quantity': label, 'value': f'{val:,.3f}', 'unit': unit})
    if res.get('pileup_pct') is not None:
        rows.append({'quantity': 'pile-up', 'value': f"{res['pileup_pct']:.2f}", 'unit': '%'})
    if res.get('count_rate_mhz') is not None:
        rows.append({'quantity': 'count rate', 'value': f"{res['count_rate_mhz']:.3f}", 'unit': 'MHz'})
    return rows

def _rows_to_frame(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=['quantity', 'value', 'unit'])

def buildApp():
    cfg = _C()
    store = {'res': None}
    ptu = pn.widgets.TextInput(name='PTU / SDT file', placeholder='/path/to/file.ptu', sizing_mode='stretch_width')
    picker = pn.widgets.FileSelector('~', file_pattern='*.*', only_files=True, height=260)
    file_card = pn.Card(picker, title='Browse files', collapsed=True, sizing_mode='stretch_width')
    model = pn.widgets.Select(name='Model', options=['discrete', 'tail', 'gaussian', 'lognormal'], value='discrete')
    nexp = pn.widgets.IntSlider(name='Exponentials', start=1, end=3, value=int(cfg['n_exp']))
    ncomp = pn.widgets.IntSlider(name='Distribution components', start=1, end=3, value=1, visible=False)
    mode = pn.widgets.Select(name='Mode', options=['summed', 'perPixel', 'both'], value=cfg['D_mode'])
    irf_source = pn.widgets.Select(name='IRF', options={'Machine IRF': 'machine', 'Estimate from data': 'estimate'}, value='machine')
    tau_min = pn.widgets.FloatInput(name='Tau min (ns)', value=float(cfg['Tau_min']), step=0.01)
    tau_max = pn.widgets.FloatInput(name='Tau max (ns)', value=float(cfg['Tau_max']), step=0.1)
    pileup = pn.widgets.Checkbox(name='Pile-up correction', value=False)
    out = pn.widgets.TextInput(name='Output name', value=cfg['OUT_NAME'])
    run = pn.widgets.Button(name='Run fit', button_type='primary', sizing_mode='stretch_width')
    status = pn.pane.Markdown('Idle.', sizing_mode='stretch_width')
    bar = pn.indicators.Progress(value=0, max=100, sizing_mode='stretch_width', visible=False)
    preview = pn.pane.Matplotlib(blank_figure('no file loaded'), dpi=110, tight=True)
    decay = pn.pane.Matplotlib(blank_figure('no fit yet'), dpi=110, tight=True)
    taumap = pn.pane.Matplotlib(blank_figure('no fit yet'), dpi=110, tight=True)
    disp_min = pn.widgets.FloatInput(name='Display min (ns)', value=0.0, step=0.1)
    disp_max = pn.widgets.FloatInput(name='Display max (ns)', value=5.0, step=0.1)
    map_key = pn.widgets.Select(name='Map', options=['tau_mean_int', 'tau_mean_amp'], value='tau_mean_int')
    table = pn.widgets.Tabulator(value=None, show_index=False, height=300, sizing_mode='stretch_width')
    log = pn.pane.Markdown('', sizing_mode='stretch_width', styles={'font-family': 'monospace', 'font-size': '11px'})
    cancel = threading.Event()

    def push(fn):
        if pn.state.curdoc is not None:
            pn.state.execute(fn)
        else:
            fn()

    def set_pane(pane, fig):
        pane.object = fig
        pane.param.trigger('object')

    def toggle_ncomp(event):
        ncomp.visible = event.new in ('gaussian', 'lognormal')
        nexp.visible = event.new in ('discrete', 'tail')
    model.param.watch(toggle_ncomp, 'value')

    def take_pick(event):
        if event.new:
            ptu.value = event.new[0]
            file_card.collapsed = True
    picker.param.watch(take_pick, 'value')

    def load_preview(path):
        try:
            set_pane(preview, intensity_figure(path))
            status.object = f'Loaded `{Path(path).name}`.'
        except Exception as exc:
            set_pane(preview, blank_figure('preview failed'))
            status.object = f'**Preview failed:** {exc}'

    def on_path(event):
        path = (event.new or '').strip()
        if path and Path(path).is_file():
            status.object = 'Building intensity preview...'
            load_preview(path)
    ptu.param.watch(on_path, 'value')

    def redraw_lifetime(event=None):
        res = store['res']
        if res is None:
            return
        set_pane(taumap, lifetime_figure(res, vmin=disp_min.value, vmax=disp_max.value, key=map_key.value))
    for w in (disp_min, disp_max, map_key):
        w.param.watch(redraw_lifetime, 'value')

    def collect():
        return {
            'ptu': ptu.value,
            'model': model.value,
            'nexp': nexp.value,
            'ncomp': ncomp.value,
            'mode': mode.value,
            'irf_source': irf_source.value,
            'tau_min': tau_min.value,
            'tau_max': tau_max.value,
            'correct_pileup': pileup.value,
            'out': out.value,
        }

    def on_progress(current, total):
        try:
            pct = int(100 * float(current) / float(total)) if total else 0
        except (TypeError, ZeroDivisionError, ValueError):
            return
        def apply():
            bar.value = max(0, min(100, pct))
            status.object = f'Fitting pixels: {current:,} / {total:,}'
        push(apply)

    def worker(a):
        from flimkit.interactive import _run_flim_fit
        try:
            res = _run_flim_fit(a, on_progress, cancel)
        except Exception:
            tb = traceback.format_exc()
            def fail():
                bar.visible = False
                run.disabled = False
                status.object = '**Fit failed.**'
                log.object = f'```\n{tb}\n```'
            push(fail)
            return
        def done():
            store['res'] = res
            bar.visible = False
            run.disabled = False
            status.object = f"Done. Output in `{Path(a.out).parent}`."
            try:
                table.value = _rows_to_frame(summary_rows(res))
            except Exception as exc:
                log.object = f'summary render failed: {exc}'
            set_pane(decay, decay_figure(res))
            lo, hi = autoscale(res, map_key.value)
            if lo is not None:
                disp_min.value = round(float(lo), 3)
                disp_max.value = round(float(hi), 3)
            set_pane(taumap, lifetime_figure(res, vmin=disp_min.value, vmax=disp_max.value, key=map_key.value))
        push(done)

    def do_run(event):
        path = ptu.value.strip()
        if not path or not Path(path).is_file():
            status.object = '**No such file.** Pick a valid PTU/SDT path.'
            return
        cancel.clear()
        run.disabled = True
        bar.value = 0
        bar.visible = True
        status.object = 'Fitting...'
        log.object = ''
        a = build_fov_args(collect())
        threading.Thread(target=worker, args=(a,), daemon=True).start()
    run.on_click(do_run)

    controls = pn.Column(
        ptu,
        file_card,
        pn.layout.Divider(),
        model, nexp, ncomp, mode, irf_source,
        pn.Row(tau_min, tau_max),
        pileup, out,
        pn.layout.Divider(),
        run, bar, status,
        width=380,
    )
    lifetime_tab = pn.Column(
        pn.Row(map_key, disp_min, disp_max),
        taumap,
    )
    results = pn.Tabs(
        ('Preview', preview),
        ('Decay', decay),
        ('Lifetime map', lifetime_tab),
        ('Summary', pn.Column(table, log)),
    )
    return pn.template.FastListTemplate(
        title='FLIMKit',
        theme='dark',
        sidebar=[controls],
        main=[results],
        sidebar_width=400,
    )

def serve(port=5006, show=True):
    pn.serve(buildApp, port=port, show=show, threaded=False)

if __name__ == '__main__':
    serve()
