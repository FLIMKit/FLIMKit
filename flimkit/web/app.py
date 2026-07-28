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

ACCENT = '#6366f1'
ACCENT_LIGHT = '#818cf8'
BG = '#0a0a0b'
PANEL = '#0d0d0f'
BORDER = '#1f1f23'
BORDER2 = '#1a1a1f'
FG = '#e4e4e7'
MUTED = '#a1a1aa'
DIM = '#71717a'
FAINT = '#52525b'

THEME_CSS = f'''
:root {{
  --panel-bg: {BG};
  --card-bg: {PANEL};
  --border: {BORDER};
}}
body, .bk-root {{ font-family: 'Inter', system-ui, sans-serif; }}
.mono, .tabulator {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}
#header, .pn-bar {{ border-bottom: 1px solid {BORDER}; }}
#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
.card, .bk-panel-models-layout-Card {{
  background: {PANEL} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}}
.bk-btn-primary, button.solid.primary {{ background: {ACCENT} !important; border-color: {ACCENT} !important; }}
.bk-btn-primary:hover {{ background: #5457e5 !important; }}
label, .bk-input-group label {{ color: {MUTED} !important; font-size: 11px; font-weight: 500; }}
input, select, .bk-input {{
  background: {BG} !important; color: {FG} !important;
  border: 1px solid #27272d !important; border-radius: 6px !important;
}}
.bk-slider-title, .noUi-connect {{ color: {ACCENT} !important; }}
.tabulator {{ background: {PANEL} !important; color: {FG} !important; border: 1px solid {BORDER} !important; }}
.tabulator .tabulator-header {{ background: {BORDER2} !important; color: {MUTED} !important; border: none !important; }}
.tabulator-row {{ background: {PANEL} !important; color: {FG} !important; border-color: {BORDER2} !important; }}
.tabulator-row.tabulator-selectable:hover {{ background: {BORDER2} !important; }}
.bk-tabs-header .bk-tab {{ color: {DIM} !important; font-size: 12px; }}
.bk-tabs-header .bk-tab.bk-active {{ color: {FG} !important; background: #16161a !important; border-radius: 6px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {FG}; }}
'''

FONT_URL = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap'

PAGE_CSS = f'''
@import url('{FONT_URL}');
html, body {{ margin: 0; height: 100%; background: {BG}; }}
.bk-root, body {{ font-family: 'Inter', system-ui, sans-serif; color: {FG}; }}
'''

SIDEBAR_CSS = f'''
:host {{
  background: {PANEL};
  border-right: 1px solid {BORDER};
  height: 100vh;
  padding: 0;
}}
'''

NAV_ITEM_CSS = f'''
:host {{ margin: 1px 0; }}
:host(.solid) .bk-btn, .bk-btn {{
  background: transparent !important;
  border: none !important;
  color: {MUTED} !important;
  text-align: left !important;
  justify-content: flex-start !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 10px !important;
  border-radius: 6px !important;
  display: flex; align-items: center; gap: 10px;
}}
.bk-btn:hover {{ background: #141418 !important; color: {FG} !important; }}
'''

NAV_ACTIVE_CSS = f'''
:host {{ margin: 1px 0; }}
.bk-btn {{
  background: #1a1a20 !important;
  border: none !important;
  color: #fafafa !important;
  text-align: left !important;
  justify-content: flex-start !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 10px !important;
  border-radius: 6px !important;
  display: flex; align-items: center; gap: 10px;
}}
.bk-btn svg, .bk-btn i {{ color: {ACCENT_LIGHT} !important; stroke: {ACCENT_LIGHT} !important; }}
'''

HEADER_CSS = f'''
:host {{
  background: {PANEL};
  border-bottom: 1px solid {BORDER};
  height: 56px;
  padding: 0 20px;
  align-items: center;
}}
'''

CARD_CSS = f'''
:host {{
  background: {PANEL};
  border: 1px solid {BORDER};
  border-radius: 8px;
  padding: 0;
  overflow: hidden;
}}
'''

def _labeled_html(text, cls):
    return pn.pane.HTML(f'<div class="{cls}">{text}</div>', margin=0,
                        sizing_mode='stretch_width')

def card(title, *objs, **kw):
    head = pn.pane.HTML(
        f'<div style="padding:10px 16px;border-bottom:1px solid {BORDER2};'
        f'font-size:12px;font-weight:500;color:#d4d4d8;">{title}</div>',
        margin=0, sizing_mode='stretch_width')
    body = pn.Column(*objs, margin=(4, 8, 8, 8), sizing_mode='stretch_width')
    return pn.Column(head, body, stylesheets=[CARD_CSS],
                     sizing_mode=kw.get('sizing_mode', 'stretch_width'),
                     margin=kw.get('margin', (0, 0, 12, 0)))

NAV_FUNCS = [
    ('Single FOV', 'scan'),
    ('ROI analysis', 'crop'),
    ('Phasor', 'circle-dot'),
    ('Stitch', 'layout-grid'),
    ('Batch', 'stack-2'),
    ('IRF builder', 'wave-sine'),
]

pn.extension('tabulator', notifications=True, raw_css=[THEME_CSS, PAGE_CSS])

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
    clipped = np.clip(img, 0, np.percentile(img, 99))
    fig = Figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(clipped, cmap='inferno', origin='upper')
    ax.set_title(f'{Path(ptu_path).name}  {img.shape[1]}x{img.shape[0]}', fontsize=8)
    ax.set_axis_off()
    style_dark(fig, ax)
    style_cbar(fig.colorbar(im, ax=ax, fraction=0.046, label='photons'))
    fig.tight_layout()
    return fig

def make_decay_bokeh():
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    src_data = ColumnDataSource(dict(t=[], y=[]))
    src_model = ColumnDataSource(dict(t=[], y=[]))
    src_resid = ColumnDataSource(dict(t=[], r=[]))
    top = figure(height=320, sizing_mode='stretch_width', y_axis_type='log',
                 background_fill_color=BG, border_fill_color=BG,
                 outline_line_color='#444', title='Summed decay')
    top.scatter('t', 'y', source=src_data, size=2, color='#aaaaaa', legend_label='data')
    top.line('t', 'y', source=src_model, line_width=2, color='#e63946', legend_label='fit')
    bot = figure(height=140, sizing_mode='stretch_width', x_range=top.x_range,
                 background_fill_color=BG, border_fill_color=BG,
                 outline_line_color='#444', title='weighted residuals')
    bot.line('t', 'r', source=src_resid, color='#457b9d')
    bot.xaxis.axis_label = 'time (ns)'
    for fg in (top, bot):
        fg.title.text_color = FG
        fg.xaxis.axis_label_text_color = FG
        fg.yaxis.axis_label_text_color = FG
        fg.xaxis.major_label_text_color = FG
        fg.yaxis.major_label_text_color = FG
        fg.xgrid.grid_line_color = '#333333'
        fg.ygrid.grid_line_color = '#333333'
    top.legend.label_text_color = FG
    top.legend.background_fill_color = BG
    top.legend.border_line_color = '#444'
    top.yaxis.axis_label = 'counts'
    return top, bot, src_data, src_model, src_resid

def update_decay_bokeh(sources, res):
    src_data, src_model, src_resid = sources
    t = res.get('time_ns')
    d = res.get('decay')
    g = res.get('global_summary') or {}
    if t is None or d is None:
        src_data.data = dict(t=[], y=[])
        src_model.data = dict(t=[], y=[])
        src_resid.data = dict(t=[], r=[])
        return
    t = np.asarray(t, dtype=float)
    d = np.clip(np.asarray(d, dtype=float), 1.0, None)
    src_data.data = dict(t=t, y=d)
    model = g.get('model')
    if model is not None:
        src_model.data = dict(t=t, y=np.clip(np.asarray(model, dtype=float), 1.0, None))
    else:
        src_model.data = dict(t=[], y=[])
    resid = g.get('residuals')
    fw = g.get('fit_window_bins')
    if resid is not None and fw is not None:
        fs, fe = int(fw[0]), int(fw[1])
        r = np.clip(np.asarray(resid, dtype=float)[fs:fe], -5, 5)
        src_resid.data = dict(t=t[fs:fe], r=r)
    else:
        src_resid.data = dict(t=[], r=[])

def lifetime_map_key(maps):
    key = next((k for k in ('tau_mean_int', 'tau_mean_amp') if k in maps), None)
    if key is None:
        key = next((k for k in maps if isinstance(maps[k], np.ndarray) and np.asarray(maps[k]).ndim == 2), None)
    return key

def make_roi_figure():
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, LinearColorMapper, BoxEditTool, Range1d
    img_src = ColumnDataSource(dict(image=[np.zeros((1, 1))], x=[0], y=[0], dw=[1], dh=[1]))
    box_src = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))
    mapper = LinearColorMapper(palette='Inferno256', low=0, high=1)
    fig = figure(height=460, sizing_mode='stretch_width', match_aspect=True,
                 background_fill_color=BG, border_fill_color=BG,
                 outline_line_color='#444', title='Draw ROI boxes (Box Edit tool)')
    fig.image(image='image', x='x', y='y', dw='dw', dh='dh', source=img_src, color_mapper=mapper)
    rects = fig.rect('x', 'y', 'width', 'height', source=box_src,
                     fill_alpha=0.18, fill_color='#4fc3f7', line_color='#4fc3f7', line_width=1.5)
    tool = BoxEditTool(renderers=[rects], num_objects=30)
    fig.add_tools(tool)
    fig.toolbar.active_drag = tool
    fig.y_range = Range1d(1, 0)
    fig.title.text_color = FG
    fig.xaxis.major_label_text_color = FG
    fig.yaxis.major_label_text_color = FG
    return fig, img_src, box_src, mapper

def load_roi_image(img_src, mapper, fig, ptu_path, channel=None):
    from bokeh.models import Range1d
    from flimkit.web.roi import intensity_map
    img = np.asarray(intensity_map(ptu_path, channel=channel), dtype=float)
    ny, nx = img.shape
    img_src.data = dict(image=[img], x=[0], y=[0], dw=[nx], dh=[ny])
    mapper.high = float(np.percentile(img, 99)) or 1.0
    mapper.low = 0.0
    fig.x_range.start, fig.x_range.end = 0, nx
    fig.y_range = Range1d(ny, 0)
    return img.shape

def boxes_from_source(box_src):
    d = box_src.data
    boxes = []
    for cx, cy, w, h in zip(d.get('x', []), d.get('y', []), d.get('width', []), d.get('height', [])):
        boxes.append((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0))
    return boxes

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
    store = {'res': None, 'doc': None}
    ptu = pn.widgets.TextInput(name='PTU / SDT file', placeholder='/path/to/file.ptu', sizing_mode='stretch_width')
    browse = pn.widgets.Button(name='Browse for file...', button_type='default', sizing_mode='stretch_width')
    picker = pn.widgets.FileSelector('~', file_pattern='*.*', only_files=True, height=420)
    pick_ok = pn.widgets.Button(name='Use selected file', button_type='primary', width=180)
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
    decay_top, decay_bot, src_data, src_model, src_resid = make_decay_bokeh()
    decay_sources = (src_data, src_model, src_resid)
    taumap = pn.pane.Matplotlib(blank_figure('no fit yet'), dpi=110, tight=True)
    disp_min = pn.widgets.FloatInput(name='Display min (ns)', value=0.0, step=0.1)
    disp_max = pn.widgets.FloatInput(name='Display max (ns)', value=5.0, step=0.1)
    map_key = pn.widgets.Select(name='Map', options=['tau_mean_int', 'tau_mean_amp'], value='tau_mean_int')
    table = pn.widgets.Tabulator(value=None, show_index=False, height=300, sizing_mode='stretch_width')
    log = pn.pane.Markdown('', sizing_mode='stretch_width', styles={'font-family': 'monospace', 'font-size': '11px'})
    nav_buttons = {}
    for label, icon in NAV_FUNCS:
        nav_buttons[label] = pn.widgets.Button(
            name=label, icon=icon, icon_size='16px',
            sizing_mode='stretch_width', stylesheets=[NAV_ITEM_CSS])
    proj_status = pn.pane.Markdown('**Project:** none open', sizing_mode='stretch_width')
    proj_files = pn.widgets.Tabulator(value=None, show_index=False, height=180,
                                      sizing_mode='stretch_width', selectable=1)
    roi_nexp = pn.widgets.IntSlider(name='Exponentials', start=1, end=3, value=int(cfg['n_exp']))
    roi_tau_min = pn.widgets.FloatInput(name='Tau min (ns)', value=float(cfg['Tau_min']), step=0.01)
    roi_tau_max = pn.widgets.FloatInput(name='Tau max (ns)', value=float(cfg['Tau_max']), step=0.1)
    roi_fig, roi_img_src, roi_box_src, roi_mapper = make_roi_figure()
    roi_load = pn.widgets.Button(name='Load image', button_type='default', width=140)
    roi_clear = pn.widgets.Button(name='Clear boxes', button_type='default', width=140)
    roi_fit = pn.widgets.Button(name='Fit ROI decay', button_type='primary', width=160)
    roi_status = pn.pane.Markdown('Load an image, then draw boxes.', sizing_mode='stretch_width')
    roi_top, roi_bot, roi_sd, roi_sm, roi_sr = make_decay_bokeh()
    roi_decay_sources = (roi_sd, roi_sm, roi_sr)
    roi_table = pn.widgets.Tabulator(value=None, show_index=False, height=260, sizing_mode='stretch_width')
    cancel = threading.Event()

    def push(fn):
        doc = store['doc'] or pn.state.curdoc
        if doc is not None:
            doc.add_next_tick_callback(fn)
        else:
            fn()

    def set_pane(pane, fig):
        pane.object = fig
        pane.param.trigger('object')

    def toggle_ncomp(event):
        ncomp.visible = event.new in ('gaussian', 'lognormal')
        nexp.visible = event.new in ('discrete', 'tail')
    model.param.watch(toggle_ncomp, 'value')

    overlay = pn.Column(
        pn.Column(
            pn.pane.HTML(f'<div style="font-size:14px;font-weight:600;color:{FG};'
                         f'padding:4px 0 10px;">Select a file</div>'),
            picker, pick_ok,
            stylesheets=[f''':host {{ background:{PANEL}; border:1px solid {BORDER};
                border-radius:10px; padding:18px; width:720px; max-width:92vw; }}'''],
        ),
        visible=False,
        stylesheets=['''
            :host { position: fixed; inset: 0; z-index: 1000;
                    background: rgba(0,0,0,0.6); align-items: center;
                    justify-content: center; }
        '''],
        sizing_mode='stretch_both',
    )

    def open_browser(event):
        overlay.visible = True
    browse.on_click(open_browser)

    def use_pick(event):
        sel = picker.value
        if sel:
            ptu.value = sel[0]
        overlay.visible = False
    pick_ok.on_click(use_pick)

    def load_preview(path):
        try:
            set_pane(preview, intensity_figure(path))
            status.object = f'Loaded `{Path(path).name}`.'
        except Exception as exc:
            set_pane(preview, blank_figure('preview failed'))
            status.object = f'**Preview failed:** {exc}'

    def refresh_project(path):
        import pandas as pd
        p = Path(path)
        folder = p.parent
        proj_status.object = f'**Project:** `{folder.name}/`  \nfile: `{p.name}`'
        rows = []
        for f in sorted(folder.glob('*')):
            if f.suffix.lower() in ('.ptu', '.sdt', '.json', '.png', '.tif', '.tiff', '.csv'):
                rows.append({'file': f.name, 'kind': f.suffix.lstrip('.')})
        proj_files.value = pd.DataFrame(rows, columns=['file', 'kind'])

    def on_path(event):
        path = (event.new or '').strip()
        if path and Path(path).is_file():
            status.object = 'Building intensity preview...'
            load_preview(path)
            refresh_project(path)
    ptu.param.watch(on_path, 'value')

    def on_proj_pick(event):
        if not event.new or proj_files.value is None:
            return
        row = event.new[0]
        try:
            name = proj_files.value.iloc[row]['file']
        except (KeyError, IndexError):
            return
        cur = Path(ptu.value.strip())
        cand = cur.parent / name
        if cand.suffix.lower() in ('.ptu', '.sdt') and cand.is_file():
            ptu.value = str(cand)
    proj_files.param.watch(on_proj_pick, 'selection')

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
            store['irf'] = res.get('irf_prompt')
            bar.visible = False
            run.disabled = False
            status.object = f"Done. Output in `{Path(a.out).parent}`."
            try:
                table.value = _rows_to_frame(summary_rows(res))
            except Exception as exc:
                log.object = f'summary render failed: {exc}'
            update_decay_bokeh(decay_sources, res)
            lo, hi = autoscale(res, map_key.value)
            if lo is not None:
                disp_min.value = round(float(lo), 3)
                disp_max.value = round(float(hi), 3)
            set_pane(taumap, lifetime_figure(res, vmin=disp_min.value, vmax=disp_max.value, key=map_key.value))
        push(done)

    def do_run(event):
        store['doc'] = pn.state.curdoc
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

    def do_roi_load(event):
        path = ptu.value.strip()
        if not path or not Path(path).is_file():
            roi_status.object = '**No file.** Pick a PTU/SDT first.'
            return
        try:
            ny, nx = load_roi_image(roi_img_src, roi_mapper, roi_fig, path)
            roi_status.object = f'Loaded `{Path(path).name}` ({nx}x{ny}). Draw boxes with the Box Edit tool.'
        except Exception as exc:
            roi_status.object = f'**Load failed:** {exc}'
    roi_load.on_click(do_roi_load)

    def do_roi_clear(event):
        roi_box_src.data = dict(x=[], y=[], width=[], height=[])
        roi_status.object = 'Boxes cleared.'
    roi_clear.on_click(do_roi_clear)

    def roi_worker(path, boxes, params):
        from flimkit.web.roi import fit_roi
        try:
            roi_decay, summary = fit_roi(path, boxes, params, irf_cached=store.get('irf'))
        except Exception as exc:
            def fail():
                roi_fit.disabled = False
                roi_status.object = f'**ROI fit failed:** {exc}'
            push(fail)
            return
        def done_roi():
            roi_fit.disabled = False
            tc = summary.get('tcspc_res', 1.0)
            t = np.arange(len(roi_decay), dtype=float) * tc * 1e9
            res_like = {'time_ns': t, 'decay': roi_decay, 'global_summary': summary}
            update_decay_bokeh(roi_decay_sources, res_like)
            roi_table.value = _rows_to_frame(summary_rows(res_like))
            roi_status.object = (f"ROI fit done: {summary.get('n_pixels', 0):,} px, "
                                 f"IRF {summary.get('irf_source', '?')}.")
        push(done_roi)

    def do_roi_fit(event):
        store['doc'] = pn.state.curdoc
        path = ptu.value.strip()
        if not path or not Path(path).is_file():
            roi_status.object = '**No file.** Pick a PTU/SDT first.'
            return
        boxes = boxes_from_source(roi_box_src)
        if not boxes:
            roi_status.object = '**No boxes drawn.** Use the Box Edit tool on the image.'
            return
        params = {'nexp': roi_nexp.value, 'tau_min': roi_tau_min.value, 'tau_max': roi_tau_max.value}
        roi_fit.disabled = True
        roi_status.object = 'Fitting ROI decay...'
        threading.Thread(target=roi_worker, args=(path, boxes, params), daemon=True).start()
    roi_fit.on_click(do_roi_fit)

    logo = pn.pane.HTML(
        f'<div style="padding:0 20px;height:56px;display:flex;align-items:center;'
        f'border-bottom:1px solid {BORDER};">'
        f'<span style="font-size:15px;font-weight:600;letter-spacing:-0.03em;color:{FG};">'
        f'FLIM<span style="color:{ACCENT};">Kit</span></span>'
        f'<span style="margin-left:8px;font-size:10px;font-weight:500;padding:2px 6px;'
        f'border-radius:4px;background:#1a1a1f;color:{DIM};">0.9.18</span></div>',
        margin=0, sizing_mode='stretch_width')

    project_panel = pn.Column(
        pn.pane.HTML(f'<div style="padding:2px;font-size:10px;font-weight:600;'
                     f'letter-spacing:0.06em;color:{FAINT};">PROJECT</div>', margin=0),
        proj_status, proj_files,
        margin=(0, 12), sizing_mode='stretch_width',
    )
    sidebar = pn.Column(
        logo,
        pn.pane.HTML(f'<div style="padding:14px 14px 6px;font-size:10px;font-weight:600;'
                     f'letter-spacing:0.06em;color:{FAINT};">FUNCTIONS</div>', margin=0),
        pn.Column(*nav_buttons.values(), margin=(0, 10)),
        pn.layout.VSpacer(),
        pn.pane.HTML(f'<div style="border-top:1px solid {BORDER};margin:8px 0;"></div>', margin=0),
        project_panel,
        stylesheets=[SIDEBAR_CSS], width=256, sizing_mode='stretch_height',
    )

    lifetime_tab = pn.Column(pn.Row(map_key, disp_min, disp_max), taumap)
    decay_tab = pn.Column(pn.pane.Bokeh(decay_top, sizing_mode='stretch_width'),
                          pn.pane.Bokeh(decay_bot, sizing_mode='stretch_width'),
                          sizing_mode='stretch_width')
    fov_tabs = pn.Tabs(
        ('Preview', preview),
        ('Decay', decay_tab),
        ('Lifetime map', lifetime_tab),
        ('Summary', pn.Column(table, log)),
    )
    fov_params = pn.Column(model, nexp, ncomp, mode, irf_source, pn.Row(tau_min, tau_max),
                           pileup, out, run, bar, status, width=300)
    fov_view = pn.Row(
        card('Parameters', fov_params, sizing_mode='fixed', margin=(0, 16, 0, 0)),
        pn.Column(fov_tabs, sizing_mode='stretch_width'),
        sizing_mode='stretch_width',
    )
    roi_params = pn.Column(roi_nexp, pn.Row(roi_tau_min, roi_tau_max),
                           pn.Row(roi_load, roi_clear, roi_fit), roi_status,
                           pn.pane.HTML(f'<div style="font-size:11px;color:{DIM};">Reuses the '
                                        f'main-fit IRF when available, else a Gaussian estimate.</div>'),
                           width=300)
    roi_view = pn.Row(
        card('Parameters', roi_params, sizing_mode='fixed', margin=(0, 16, 0, 0)),
        pn.Column(
            card('Draw ROI boxes', pn.pane.Bokeh(roi_fig, sizing_mode='stretch_width')),
            card('ROI decay fit',
                 pn.pane.Bokeh(roi_top, sizing_mode='stretch_width'),
                 pn.pane.Bokeh(roi_bot, sizing_mode='stretch_width'),
                 roi_table),
            sizing_mode='stretch_width',
        ),
        sizing_mode='stretch_width',
    )

    def placeholder(name):
        return pn.pane.HTML(
            f'<div style="padding:40px;color:{DIM};"><div style="font-size:18px;'
            f'font-weight:600;color:{FG};margin-bottom:8px;">{name}</div>'
            f'Not yet ported to the web UI. Available in the tkinter GUI '
            f'(<code>flimkit-gui</code>).</div>', sizing_mode='stretch_width')

    views = {
        'Single FOV': fov_view,
        'ROI analysis': roi_view,
        'Phasor': placeholder('Phasor analysis'),
        'Stitch': placeholder('Tile stitch / fit'),
        'Batch': placeholder('Batch processing'),
        'IRF builder': placeholder('Machine IRF builder'),
    }
    titles = {
        'Single FOV': 'Single FOV fit', 'ROI analysis': 'ROI analysis',
        'Phasor': 'Phasor analysis', 'Stitch': 'Tile stitch / fit',
        'Batch': 'Batch processing', 'IRF builder': 'Machine IRF builder',
    }
    page_title = pn.pane.HTML('', margin=0)
    main_area = pn.Column(views['Single FOV'], margin=(16, 20), sizing_mode='stretch_both')

    def select_nav(label):
        for lbl, btn in nav_buttons.items():
            btn.stylesheets = [NAV_ACTIVE_CSS if lbl == label else NAV_ITEM_CSS]
        page_title.object = (f'<span style="font-size:15px;font-weight:600;'
                             f'letter-spacing:-0.02em;color:{FG};">{titles[label]}</span>')
        main_area[:] = [views[label]]

    for label, btn in nav_buttons.items():
        btn.on_click(lambda event, lb=label: select_nav(lb))
    select_nav('Single FOV')

    header = pn.Row(page_title, pn.layout.HSpacer(), browse,
                    stylesheets=[HEADER_CSS], sizing_mode='stretch_width', height=56)
    main = pn.Column(header, main_area, sizing_mode='stretch_both',
                     styles={'background': BG})
    page = pn.Row(sidebar, main, overlay, sizing_mode='stretch_both',
                  styles={'background': BG, 'gap': '0'})
    return page

def serve(port=5006, show=True):
    pn.serve(buildApp, port=port, show=show, threaded=False)

if __name__ == '__main__':
    serve()
