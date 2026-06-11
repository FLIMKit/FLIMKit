from flimkit.UI.roi_tools import RoiAnalysisPanel


class ModeController:
    def __init__(self, builder):
        self.b = builder

    def switch(self, form_id: str):
        for btn in self.b._form_buttons.values():
            btn.state(['!pressed'])

        if form_id in self.b._form_inner_frames:
            # batch and irf are menu-only - no sidebar button to highlight

            self.b._current_form = form_id

            if form_id == 'fov':
                self.b._stitch_tabs.grid_remove()
                if 'phasor' in self.b._form_inner_frames:
                    self.b._form_inner_frames['phasor'][0].grid_remove()
                if hasattr(self.b, '_roi_analysis_panel'):
                    self.b._fov_preview._roi_analysis_panel = self.b._roi_analysis_panel

                fov_frame = self.b._form_inner_frames['fov'][0]
                fov_frame.grid(row=0, column=0, sticky='nsew')
                fov_frame.lift()
                fov_frame.tkraise()

                self.b._analysis_tabs.grid(row=0, column=0, sticky='nsew')
                self.b._analysis_tabs.lift()
                self.b._analysis_tabs.tkraise()
                self.b._analysis_tabs.select(0)

                # update_idletasks avoids re-entrant event processing
                self.b._fit_settings_tab.update_idletasks()
                self.b._refresh_scrollable_frame(form_id)

            elif form_id == 'stitch':
                self.b._analysis_tabs.grid_remove()
                if 'phasor' in self.b._form_inner_frames:
                    self.b._form_inner_frames['phasor'][0].grid_remove()

                if 'fov' in self.b._form_inner_frames:
                    self.b._form_inner_frames['fov'][0].grid_remove()

                stitch_frame = self.b._form_inner_frames['stitch'][0]
                stitch_frame.grid(row=0, column=0, sticky='nsew')
                stitch_frame.lift()
                stitch_frame.tkraise()

                self.b._stitch_tabs.grid(row=0, column=0, sticky='nsew')
                self.b._stitch_tabs.lift()
                self.b._stitch_tabs.tkraise()

                if not hasattr(self.b, '_stitch_roi_panel'):
                    self.b._stitch_roi_panel = RoiAnalysisPanel(self.b._stitch_roi_analysis_frame)
                    self.b._stitch_roi_panel.grid(row=0, column=0, sticky='nsew')
                    self.b._stitch_roi_panel.fov_preview = self.b._fov_preview
                    self.b._fov_preview._roi_analysis_panel = self.b._stitch_roi_panel

                self.b._stitch_tabs.select(0)

                # update_idletasks avoids re-entrant event processing
                self.b._stitch_settings_tab.update_idletasks()
                self.b._refresh_scrollable_frame(form_id)

            else:
                self.b._analysis_tabs.grid_remove()
                self.b._stitch_tabs.grid_remove()

                for fid in ('phasor', 'fov', 'stitch', 'batch', 'irf'):
                    if fid != form_id and fid in self.b._form_inner_frames:
                        self.b._form_inner_frames[fid][0].grid_remove()

                if form_id in self.b._form_inner_frames:
                    selected_frame = self.b._form_inner_frames[form_id][0]
                    selected_frame.grid(row=0, column=0, sticky='nsew')
                    selected_frame.lift()
                    selected_frame.tkraise()
                    self.b._refresh_scrollable_frame(form_id)

            if form_id == 'phasor':
                self.b._fov_preview.frame.grid_remove()
                self.b._phasor_panel.frame.grid()
                self.b._preview_frame_label.configure(text='  Phasor Analysis  ')
                if (hasattr(self.b, 'sv_ph_ptu') and hasattr(self.b, 'sv_ptu')
                        and not self.b.sv_ph_ptu.get().strip()):
                    fov_ptu = self.b.sv_ptu.get().strip()
                    if fov_ptu:
                        self.b.sv_ph_ptu.set(fov_ptu)
                    if hasattr(self.b, '_irf_fov'):
                        method = self.b._irf_fov.sv_method.get()
                        if method == 'irf_xlsx' and hasattr(self.b, 'sv_xlsx'):
                            xlsx = self.b.sv_xlsx.get().strip()
                            if xlsx and not self.b.sv_ph_irf.get().strip():
                                self.b.sv_ph_irf.set(xlsx)
                        elif method == 'machine_irf':
                            mirf = self.b._irf_fov.sv_path.get().strip()
                            if mirf and not self.b.sv_ph_mirf.get().strip():
                                self.b.sv_ph_mirf.set(mirf)
            elif form_id in ('batch', 'irf'):
                self.b._phasor_panel.frame.grid_remove()
                self.b._fov_preview.frame.grid_remove()
                label = '  Machine IRF Builder  ' if form_id == 'irf' else '  Batch Processing  '
                self.b._preview_frame_label.configure(text=label)
                # Show the IRF plot canvas if it exists (created after first build)
                if form_id == 'irf' and hasattr(self.b, '_irf_plot_frame'):
                    self.b._irf_plot_frame.grid()
            else:
                self.b._phasor_panel.frame.grid_remove()
                if hasattr(self.b, '_irf_plot_frame'):
                    self.b._irf_plot_frame.grid_remove()
                self.b._fov_preview.frame.grid()

        if hasattr(self.b, 'mode_status'):
            self.b.mode_status.set(f"Current: {self.b._form_labels.get(form_id, form_id)}")
            self.b._preview_frame_label.configure(text='  FOV Preview  ')
