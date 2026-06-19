import argparse
from pathlib import Path
from flimkit.UI.utils import _C, _flt, _thresh

class FLIMKitController:
    def __init__(self, builder):
        self.b = builder

    def fov_args(self):
        cfg = _C()
        ptu = self.b.sv_ptu.get().strip()
        irf = self.b._irf_fov.get_args(xlsx_fallback=self.b.sv_xlsx.get().strip())
        a = argparse.Namespace()
        a.ptu = ptu
        a.xlsx = self.b.sv_xlsx.get().strip() or None
        a.debug_xlsx = False
        a.print_config = False
        a.irf = irf['irf']
        a.irf_xlsx = irf['irf_xlsx']
        a.estimate_irf = irf['estimate_irf']
        a.no_xlsx_irf = irf['no_xlsx_irf']
        a.machine_irf = irf.get('machine_irf') or str(_C()['MACHINE_IRF_DEFAULT_PATH'])
        a.irf_bins = cfg['IRF_BINS']
        a.irf_fit_width = cfg['IRF_FIT_WIDTH']
        a.irf_fwhm = cfg['IRF_FWHM']
        _model = self.b.sv_fit_model_fov.get()
        a.dist_type = _model
        if _model == 'discrete':
            a.nexp = self.b.iv_nexp_fov.get()
            a.dist_n_components = 1
        else:
            a.nexp = 2
            a.dist_n_components = self.b.iv_ncomp_dist_fov.get()
        a.tau_min = float(self.b.sv_tau_min_fov.get() or cfg['Tau_min'])
        a.tau_max = float(self.b.sv_tau_max_fov.get() or cfg['Tau_max'])
        a.mode = self.b.sv_mode_fov.get()
        a.binning = cfg['binning_factor']
        a.min_photons = cfg['MIN_PHOTONS_PERPIX']
        a.optimizer = cfg['Optimizer']
        a.restarts = cfg['lm_restarts']
        a.de_population = cfg['de_population']
        a.de_maxiter = cfg['de_maxiter']
        a.workers = cfg['n_workers']
        a.no_polish = False
        a.channel = cfg['channels']
        _out_raw = self.b.sv_out_fov.get().strip() or cfg['OUT_NAME']
        if Path(_out_raw).parent == Path('.'):
            a.out = str(Path(ptu).parent / _out_raw)
        else:
            a.out = _out_raw
        a.no_plots = False
        a.cell_mask = self.b.bv_cell.get()
        a.correct_pileup = self.b.bv_correct_pileup.get()
        a.intensity_threshold = _thresh(self.b.bv_thr_fov, self.b.sv_thr_fov)
        a.irf_align = 'steepest_rise'
        a.irf_shift_bins = 2
        a.tvb_ptu = (self.b.sv_tvb_ptu_fov.get().strip() or None) if hasattr(self.b, 'sv_tvb_ptu_fov') else None
        a.tvb_channel = None
        self.b._apply_expert_overrides(a)
        return a

    def stitch_args(self):
        xlif = self.b.sv_xlif.get().strip()
        ptu_dir = self.b.sv_ptu_dir.get().strip()
        out_base = self.b.sv_out_st.get().strip()
        pipeline = self.b.sv_pipeline.get()
        roi_name = Path(xlif).stem.replace(' ', '_')
        output_dir = str(Path(out_base) / roi_name)
        a = argparse.Namespace()
        a.xlif = xlif
        a.ptu_dir = ptu_dir
        a.output_dir = output_dir
        a.ptu_basename = Path(xlif).stem
        a.rotate_tiles = self.b.bv_rotate.get()
        cfg = _C()
        irf = self.b._irf_st.get_args()
        a.irf = irf['irf']
        a.irf_xlsx = irf['irf_xlsx']
        a.no_xlsx_irf = irf['no_xlsx_irf']
        a.estimate_irf = irf['estimate_irf'] if irf['estimate_irf'] != 'none' else 'gaussian'
        a.machine_irf = irf.get('machine_irf') or str(cfg['MACHINE_IRF_DEFAULT_PATH'])
        _model_st = self.b.sv_fit_model_st.get()
        a.dist_type = _model_st
        if _model_st == 'discrete':
            a.nexp = self.b.iv_nexp_st.get()
            a.dist_n_components = 1
        else:
            a.nexp = 2
            a.dist_n_components = self.b.iv_ncomp_dist_st.get()
        a.tau_min = float(self.b.sv_tau_fit_lo.get() or cfg['Tau_min'])
        a.tau_max = float(self.b.sv_tau_fit_hi.get() or cfg['Tau_max'])
        a.register_tiles = self.b.bv_register.get()
        a.reg_max_shift_px = int(self.b.sv_reg_max_shift.get() or 120)
        a.binning = cfg['binning_factor']
        a.min_photons = cfg['MIN_PHOTONS_PERPIX']
        a.optimizer = 'de'
        a.restarts = cfg['lm_restarts']
        a.de_population = cfg['de_population']
        a.de_maxiter = cfg['de_maxiter']
        a.workers = cfg['n_workers']
        a.no_polish = False
        a.channel = cfg['channels']
        a.irf_fwhm = cfg['IRF_FWHM']
        a.irf_bins = cfg['IRF_BINS']
        a.irf_fit_width = cfg['IRF_FIT_WIDTH']
        a.tau_display_min = _flt(self.b.sv_tau_lo)
        a.tau_display_max = _flt(self.b.sv_tau_hi)
        a.intensity_display_min = _flt(self.b.sv_int_lo)
        a.intensity_display_max = _flt(self.b.sv_int_hi)
        a.intensity_threshold = _thresh(self.b.bv_thr_st, self.b.sv_thr_st)
        a.correct_pileup = self.b.bv_correct_pileup_st.get()
        a.save_individual = self.b.bv_save_ind.get()
        a.save_tau_weighted = self.b.bv_save_tau_weighted.get()
        a.save_int_weighted = self.b.bv_save_int_weighted.get()
        a.save_amp_weighted = self.b.bv_save_amp_weighted.get()
        a.irf_align = 'steepest_rise'
        a.irf_shift_bins = 2
        a.tvb_ptu = (self.b.sv_tvb_ptu_st.get().strip() or None) if hasattr(self.b, 'sv_tvb_ptu_st') else None
        a.tvb_channel = None
        self.b._apply_expert_overrides(a)
        if pipeline == 'tile_fit':
            a.mode = 'both'
            a.no_plots = True
            a.cell_mask = False
            a.debug_xlsx = False
            a.print_config = False
            a.xlsx = None
            a.out = None
            a.irf_xlsx_dir = self.b.sv_tile_irf_dir.get().strip() or None
        else:
            a.mode = 'both' if self.b.bv_perpix.get() else 'summed'
            a.no_plots = False
            a.irf_xlsx_dir = None
        return a
