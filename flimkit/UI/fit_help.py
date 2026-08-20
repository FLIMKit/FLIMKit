from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

TOPIC_ORDER = ('fit_model', 'components', 'fitting_mode', 'optimizer',
               'irf', 'masking')

FIT_HELP = {
    'fit_model': ('Fit model', [
        ('n-exp',
         'A sum of N discrete exponentials, convolved with the IRF before it is '
         'compared to the data. This is the default and the right choice for most '
         'samples. It assumes every molecule in a pixel sits in one of N '
         'well-defined states, each with a single lifetime.'),
        ('Gaussian dist.',
         'One continuous distribution of lifetimes instead of a discrete set. The '
         'fit returns a centre and a width rather than a list of taus. Use it when '
         'the fluorophore sits in a heterogeneous environment and a discrete fit '
         'needs an implausible number of components to converge.'),
        ('Lorentzian dist.',
         'The same idea with a Lorentzian shape, which has heavier tails than a '
         'Gaussian. It tolerates a small sub-population well away from the centre '
         'without dragging the centre towards it.'),
        ('n-exp tail',
         'Fits discrete exponentials past the peak of the decay and uses no IRF at '
         'all. Fast, and it removes the IRF as a source of error, but it cannot '
         'recover components shorter than roughly the IRF width, so short taus come '
         'out biased. The IRF panel disappears when this is selected.'),
    ], 'Distribution fits are slower than discrete ones: each evaluation integrates '
       'over a 200-point lifetime grid.'),

    'components': ('Components', [
        ('n-exp models (1, 2, 3)',
         'How many discrete exponentials to fit. Start at 1 and add a component '
         'only if the residuals show structure. Each extra component adds two free '
         'parameters, and three components need a lot of photons before the taus '
         'are separable.'),
        ('Distribution models (unimodal, bimodal)',
         'How many distributions to sum. Unimodal is one centre and one width. '
         'Bimodal fits two, for a sample with two distinct populations that are '
         'each internally heterogeneous.'),
    ], 'A lower chi-squared alone does not justify an extra component. More '
       'parameters will always fit better. Check whether the recovered taus are '
       'physically distinct and whether the residuals actually improved.'),

    'fitting_mode': ('Fitting mode', [
        ('Full',
         'Fits the summed decay for the whole field of view and then fits every '
         'pixel individually. This is what produces the FLIM image and the lifetime '
         'maps.'),
        ('Fast',
         'Fits the summed decay only. No per-pixel fitting, so no FLIM image and no '
         'lifetime maps, but it finishes in seconds rather than minutes. Use it to '
         'check that the IRF, the model and the component count are sensible before '
         'committing to a full run.'),
    ], 'Per-pixel fitting is the expensive step. Binning and the minimum '
       'photons-per-pixel threshold both live in Expert Fit Settings.'),

    'optimizer': ('Optimizer', [
        ('Differential Evolution (DE)',
         'A global search over the whole bounded parameter space, so it does not '
         'depend on a starting guess and will not settle into a local minimum near '
         'one. Slower than LM. Lifetimes are searched in log space, the population '
         'is Sobol-initialised, and a Levenberg-Marquardt polish is run at the end '
         'to sharpen the result.'),
        ('Levenberg-Marquardt (LM)',
         'A local gradient-based fit, run from several starting points and the best '
         'result kept. The first start is log-spaced across the lifetime bounds and '
         'the rest are random. Much faster than DE, but with few restarts it can '
         'still miss the global minimum on a multi-exponential decay.'),
    ], 'Both are seeded, so repeating a fit on the same data gives the same answer. '
       'DE population, DE iterations and LM restart count are all set in Expert Fit '
       'Settings.'),

    'irf': ('Instrument Response Function (IRF)', [
        ('Analytical model (LAS X export)',
         'Builds the IRF from the LAS X export named in Input Files. This is the '
         'default for Leica FALCON data, where the microscope has already '
         'characterised its own response.'),
        ('Machine IRF (.npy pre-built)',
         'A prompt measured once for the instrument and saved to disk, reused for '
         'every dataset from that microscope. Build one with the IRF tab. Use this '
         'when the optical path has not changed since it was measured.'),
        ('Machine IRF + full sigma broadening',
         'The same stored prompt, but the fit is allowed to broaden it with a '
         'Gaussian of up to 3.0 bins. Use it when the stored IRF is narrower than '
         'the real response, for example after a change of objective or pinhole.'),
        ('Machine IRF + half sigma broadening',
         'Broadening capped at 0.5 bins. A tighter leash for when the stored IRF is '
         'close to right and a free sigma would start absorbing the decay itself.'),
        ('Measured IRF file (scatter PTU or .pck)',
         'A prompt measured alongside this dataset, usually a scattering solution '
         'or a reflecting surface. The most defensible option when you have one: it '
         'carries the same optical path, detector and timing as the sample.'),
        ('Estimate from decay - raw',
         'Takes 21 bins centred on the peak of the decay itself and uses them as '
         'the prompt, after subtracting a background estimated from the pre-peak '
         'bins. A last resort. It cannot separate the instrument response from the '
         'fastest part of the decay, so short lifetimes come out too long.'),
        ('Estimate from decay - parametric',
         'Fits an analytical pulse shape, amplitude times t/t0 times exp(-t/t0), to '
         'a 1.5 ns window around the peak. Smoother and less noisy than raw '
         'extraction, but it inherits the same problem and falls back to raw '
         'extraction if the fit fails.'),
        ('Gaussian (fallback)',
         'A plain Gaussian prompt of the configured width. For when nothing better '
         'is available and you need a number rather than an answer.'),
    ], 'The IRF is the largest single source of systematic error in a reconvolution '
       'fit, and it matters most for the shortest component. If tau_1 comes out '
       'near the IRF width, treat it as unresolved rather than measured. Picking '
       'the n-exp tail model removes the IRF from the fit entirely.'),

    'masking': ('Masking & Thresholding', [
        ('Apply cell mask (Cellpose-SAM)',
         'Segments cells in the intensity image and fits only inside them. Removes '
         'background pixels that would otherwise contribute noise-dominated '
         'lifetimes to the summary statistics.'),
        ('Intensity threshold (min photons/px)',
         'Skips any pixel with fewer than this many photons. A per-pixel fit on a '
         'few dozen photons is dominated by shot noise, so the resulting lifetime '
         'is close to meaningless and will smear the histogram. Leave blank for no '
         'threshold.'),
        ('Coates pile-up correction',
         'At high count rates the detector preferentially records early photons, '
         'which biases lifetimes short. The Coates correction inverts that '
         'distortion on the measured decay. The count rate and photons-per-pulse '
         'are printed at the start of every fit, and the checkbox is worth ticking '
         'above about 5 percent.'),
        ('Time-varying background PTU',
         'A separately measured background acquisition. Its decay shape is fitted '
         'as an extra component with a free scale, so a background that is itself '
         'time-varying, such as detector afterpulsing or room light, is subtracted '
         'in shape rather than as a flat offset.'),
    ], 'The Coates correction rescales the counts, so the corrected decay is no '
       'longer Poisson-distributed while the default cost function still assumes it '
       'is. The reported chi-squared is unreliable when the correction is on, even '
       'though the recovered lifetimes are not.'),
}


class FitHelpWindow(tk.Toplevel):

    def __init__(self, parent, topic=None):
        super().__init__(parent)
        self.title('Choosing your fit settings')
        self.transient(parent)
        self.geometry('620x520')
        self.minsize(420, 300)
        frame = ttk.Frame(self, padding=(12, 10))
        frame.pack(fill='both', expand=True)
        self._text = tk.Text(frame, wrap='word', relief='flat', padx=8, pady=6,
                             borderwidth=0, highlightthickness=0)
        bar = ttk.Scrollbar(frame, orient='vertical', command=self._text.yview)
        self._text.configure(yscrollcommand=bar.set)
        self._text.pack(side='left', fill='both', expand=True)
        bar.pack(side='right', fill='y')
        self._configure_tags()
        self._marks = {}
        self._fill()
        self._text.configure(state='disabled')
        ttk.Button(self, text='Close', command=self.destroy).pack(pady=(0, 10))
        self.bind('<Escape>', lambda _e: self.destroy())
        if topic in self._marks:
            self.after(50, lambda: self._text.see(self._marks[topic]))

    def _configure_tags(self):
        base = tkfont.nametofont('TkDefaultFont')
        size = base.cget('size')
        family = base.cget('family')
        self._text.tag_configure('h1', font=(family, abs(size) + 3, 'bold'),
                                 spacing1=14, spacing3=6)
        self._text.tag_configure('h2', font=(family, abs(size), 'bold'),
                                 spacing1=8, spacing3=2, lmargin1=8, lmargin2=8)
        self._text.tag_configure('body', spacing3=4, lmargin1=8, lmargin2=8)
        self._text.tag_configure('note', foreground='#777', spacing1=6, spacing3=8,
                                 lmargin1=8, lmargin2=8)

    def _fill(self):
        for key in TOPIC_ORDER:
            title, entries, note = FIT_HELP[key]
            self._marks[key] = f'mark_{key}'
            self._text.mark_set(self._marks[key], 'end-1c')
            self._text.mark_gravity(self._marks[key], 'left')
            self._text.insert('end', title + '\n', 'h1')
            for name, body in entries:
                self._text.insert('end', name + '\n', 'h2')
                self._text.insert('end', body + '\n', 'body')
            self._text.insert('end', note + '\n', 'note')


def show_fit_help(parent, topic=None):
    win = FitHelpWindow(parent, topic=topic)
    win.focus_set()
    return win


def help_button(parent, topic):
    btn = ttk.Label(parent, text='ⓘ', foreground='#4a7ebb', cursor='hand2',
                    takefocus=True)
    def _open(_evt=None):
        show_fit_help(parent.winfo_toplevel(), topic)
    for seq in ('<Button-1>', '<Return>', '<space>'):
        btn.bind(seq, _open)
    return btn
