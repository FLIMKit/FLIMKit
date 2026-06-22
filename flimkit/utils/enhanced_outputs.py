import numpy as np
import tifffile
from pathlib import Path


def save_fit_summary_txt(
    summary,
    output_path,
    n_exp=2,
    strategy='gaussian',
    metadata=None
):
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('FLIM FIT RESULTS SUMMARY\n')
        f.write('='*60 + '\n\n')
        
        # Metadata
        if metadata:
            f.write('ROI Information:\n')
            f.write('-'*60 + '\n')
            if 'canvas_shape' in metadata:
                f.write(f"Canvas size: {metadata['canvas_shape'][0]} × {metadata['canvas_shape'][1]} pixels\n")
            if 'tiles_processed' in metadata:
                f.write(f"Tiles processed: {metadata['tiles_processed']}\n")
            if 'total_photons' in metadata:
                f.write(f"Total photons: {metadata['total_photons']:,.0f}\n")
            f.write('\n')
        
        # Fit parameters
        f.write('Fit Parameters:\n')
        f.write('-'*60 + '\n')
        f.write(f"Number of exponentials: {n_exp}\n")
        f.write(f"IRF strategy: {strategy}\n")
        f.write(f"Chi-squared (reduced): {summary.get('chi2r', 0):.6f}\n")
        f.write(f"Background: {summary.get('bg', 0):.3f}\n")
        if summary.get('tvb_scale', 0):
            f.write(f"TVB scale: {summary['tvb_scale']:.3f}\n")
        if 'sigma' in summary:
            f.write(f"IRF broadening (σ): {summary['sigma']:.3f} ns\n")
        f.write('\n')
        
        # Lifetime components
        f.write('Lifetime Components:\n')
        f.write('-'*60 + '\n')
        for i in range(1, n_exp + 1):
            tau_key = f'tau_{i}'
            amp_key = f'a{i}'
            
            if tau_key in summary:
                tau = summary[tau_key]
                f.write(f"Component {i}:\n")
                f.write(f"  τ{i} = {tau:.4f} ns\n")
                
                if amp_key in summary:
                    amp = summary[amp_key]
                    f.write(f"  A{i} = {amp:.4f}\n")
                    
                    # Calculate fractional intensity
                    total_amp = sum(summary.get(f'a{j}', 0) for j in range(1, n_exp + 1))
                    if total_amp > 0:
                        frac = amp / total_amp * 100
                        f.write(f"  Fractional intensity: {frac:.1f}%\n")
                
                f.write('\n')
        
        # Average lifetime
        if n_exp > 1:
            tau_avg = 0
            total_amp = 0
            for i in range(1, n_exp + 1):
                tau = summary.get(f'tau_{i}', 0)
                amp = summary.get(f'a{i}', 0)
                tau_avg += tau * amp
                total_amp += amp
            
            if total_amp > 0:
                tau_avg /= total_amp
                f.write(f"Average lifetime (amplitude-weighted): {tau_avg:.4f} ns\n\n")
        
        # All parameters (raw)
        f.write('Raw Parameters:\n')
        f.write('-'*60 + '\n')
        for key, value in sorted(summary.items()):
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.6e}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write('\n' + '='*60 + '\n')
    
    print(f"Fit summary saved: {output_path}")


def save_weighted_tau_images(
    pixel_maps,
    output_dir,
    roi_name='ROI',
    n_exp=2,
    save_intensity=True,
    save_amplitude=True,
    tau_display_min=None,
    tau_display_max=None,
    intensity_display_min=None,
    intensity_display_max=None,
    target_shape=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # stitch_flim_tiles saves *_stitched_intensity.tif at full (H, W).
    # fit_per_pixel with binning>1 returns maps at (H//b, W//b).
    # Nearest-neighbour resize corrects the mismatch without blurring values.
    if target_shape is not None:
        th, tw = int(target_shape[0]), int(target_shape[1])
        try:
            import cv2 as _cv2
            def _up(a):
                if a is None or a.ndim < 2 or a.shape[:2] == (th, tw): return a
                return _cv2.resize(a.astype(np.float32), (tw, th),
                                   interpolation=_cv2.INTER_NEAREST)
        except ImportError:
            def _up(a):
                if a is None or a.ndim < 2 or a.shape[:2] == (th, tw): return a
                h, w = a.shape[:2]
                rh, rw = max(1, round(th / h)), max(1, round(tw / w))
                return np.repeat(np.repeat(a, rh, axis=0), rw, axis=1)[:th, :tw]
        pixel_maps = {k: _up(v) for k, v in pixel_maps.items()}
        print(f"  \u2191 Pixel maps upsampled to {th}\xd7{tw} px")
    # Get intensity map (total photon counts per pixel)
    if 'intensity' in pixel_maps:
        intensity = pixel_maps['intensity']
    else:
        # Calculate from amplitudes if not provided
        intensity = np.zeros_like(pixel_maps['tau_1'])
        for i in range(1, n_exp + 1):
            amp_key = f'a{i}'
            if amp_key in pixel_maps:
                intensity += pixel_maps[amp_key]
    
    # Save intensity image - scaled to full uint16 range like stitch-only
    if save_intensity:
        intensity_out = intensity.copy().astype(np.float64)
        # Apply intensity display range (clip to boundaries, FLIM microscope style)
        if intensity_display_min is not None:
            intensity_out = np.where(
                intensity_out > 0,
                np.clip(intensity_out, intensity_display_min, intensity_out),
                intensity_out,
            )
        if intensity_display_max is not None:
            intensity_out = np.where(
                intensity_out > 0,
                np.clip(intensity_out, intensity_out, intensity_display_max),
                intensity_out,
            )
        intensity_path = output_dir / f"{roi_name}_intensity.tif"
        max_val = intensity_out.max()
        if max_val > 0:
            intensity_scaled = (intensity_out / max_val * 65535).astype(np.uint16)
        else:
            intensity_scaled = np.zeros_like(intensity_out, dtype=np.uint16)
        tifffile.imwrite(str(intensity_path), intensity_scaled)
        print(f"Intensity image saved: {intensity_path} (uint16, max-scaled)")
        if intensity_display_min is not None or intensity_display_max is not None:
            lo = intensity_display_min if intensity_display_min is not None else 'auto'
            hi = intensity_display_max if intensity_display_max is not None else 'auto'
            print(f"  Intensity display range: [{lo}, {hi}] (clipped)")
    
    # Compute intensity-weighted average lifetime
    if save_intensity and n_exp > 1:
        tau_intensity_weighted = np.zeros_like(intensity, dtype=np.float32)
        
        for i in range(1, n_exp + 1):
            tau_key = f'tau_{i}'
            amp_key = f'a{i}'
            
            if tau_key in pixel_maps and amp_key in pixel_maps:
                tau = pixel_maps[tau_key]
                amp = pixel_maps[amp_key]
                
                # Weight by amplitude (which represents intensity contribution)
                tau_intensity_weighted += tau * amp
        
        # Normalize by total intensity
        mask = intensity > 0
        tau_intensity_weighted[mask] /= intensity[mask]
        tau_intensity_weighted[~mask] = 0
        
        # Apply lifetime display range (clip to boundaries, FLIM microscope style)
        if tau_display_min is not None or tau_display_max is not None:
            lo = tau_display_min if tau_display_min is not None else tau_intensity_weighted[mask].min() if mask.any() else 0
            hi = tau_display_max if tau_display_max is not None else tau_intensity_weighted[mask].max() if mask.any() else 0
            tau_intensity_weighted[mask] = np.clip(tau_intensity_weighted[mask], lo, hi)
        
        # Save
        tau_int_path = output_dir / f"{roi_name}_tau_intensity_weighted.tif"
        tifffile.imwrite(str(tau_int_path), tau_intensity_weighted)
        print(f"Intensity-weighted tau image saved: {tau_int_path}")
        if mask.any():
            print(f"  Range: {tau_intensity_weighted[mask].min():.3f} - {tau_intensity_weighted[mask].max():.3f} ns")
            if tau_display_min is not None or tau_display_max is not None:
                lo_s = f"{tau_display_min}" if tau_display_min is not None else 'auto'
                hi_s = f"{tau_display_max}" if tau_display_max is not None else 'auto'
                print(f"  Tau display range: [{lo_s}, {hi_s}] ns (clipped)")
        else:
            print(f"  Range: no valid pixels")
    
    # Compute amplitude-weighted average lifetime
    if save_amplitude and n_exp > 1:
        tau_amplitude_weighted = np.zeros_like(intensity, dtype=np.float32)
        total_amplitude = np.zeros_like(intensity, dtype=np.float32)
        
        for i in range(1, n_exp + 1):
            tau_key = f'tau_{i}'
            amp_key = f'a{i}'
            
            if tau_key in pixel_maps and amp_key in pixel_maps:
                tau = pixel_maps[tau_key]
                amp = pixel_maps[amp_key]
                
                # Weight by amplitude (fractional contribution)
                tau_amplitude_weighted += tau * amp
                total_amplitude += amp
        
        # Normalize by total amplitude
        mask = total_amplitude > 0
        tau_amplitude_weighted[mask] /= total_amplitude[mask]
        tau_amplitude_weighted[~mask] = 0
        
        # Apply lifetime display range (clip to boundaries, FLIM microscope style)
        if tau_display_min is not None or tau_display_max is not None:
            lo = tau_display_min if tau_display_min is not None else tau_amplitude_weighted[mask].min() if mask.any() else 0
            hi = tau_display_max if tau_display_max is not None else tau_amplitude_weighted[mask].max() if mask.any() else 0
            tau_amplitude_weighted[mask] = np.clip(tau_amplitude_weighted[mask], lo, hi)
        
        # Save
        tau_amp_path = output_dir / f"{roi_name}_tau_amplitude_weighted.tif"
        tifffile.imwrite(str(tau_amp_path), tau_amplitude_weighted)
        print(f"Amplitude-weighted tau image saved: {tau_amp_path}")
        if mask.any():
            print(f"  Range: {tau_amplitude_weighted[mask].min():.3f} - {tau_amplitude_weighted[mask].max():.3f} ns")
            if tau_display_min is not None or tau_display_max is not None:
                lo_s = f"{tau_display_min}" if tau_display_min is not None else 'auto'
                hi_s = f"{tau_display_max}" if tau_display_max is not None else 'auto'
                print(f"  Tau display range: [{lo_s}, {hi_s}] ns (clipped)")
        else:
            print(f"  Range: no valid pixels")
    
    # For single exponential, just save the single tau map
    if n_exp == 1:
        tau_single = pixel_maps['tau_1']
        tau_path = output_dir / f"{roi_name}_tau.tif"
        tifffile.imwrite(str(tau_path), tau_single.astype(np.float32))
        print(f"Lifetime image saved: {tau_path}")
        mask = tau_single > 0
        if mask.any():
            print(f"  Range: {tau_single[mask].min():.3f} - {tau_single[mask].max():.3f} ns")
        else:
            print(f"  Range: no valid pixels")


def save_individual_tau_maps(
    pixel_maps,
    output_dir,
    roi_name='ROI',
    n_exp=2
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(1, n_exp + 1):
        # Save tau map
        tau_key = f'tau_{i}'
        if tau_key in pixel_maps:
            tau = pixel_maps[tau_key]
            tau_path = output_dir / f"{roi_name}_tau{i}.tif"
            tifffile.imwrite(str(tau_path), tau.astype(np.float32))
            print(f"τ{i} map saved: {tau_path}")
        
        # Save amplitude map
        amp_key = f'a{i}'
        if amp_key in pixel_maps:
            amp = pixel_maps[amp_key]
            amp_path = output_dir / f"{roi_name}_a{i}.tif"
            tifffile.imwrite(str(amp_path), amp.astype(np.float32))
            print(f"A{i} map saved: {amp_path}")

    if 'tvb_scale' in pixel_maps:
        tvb_path = output_dir / f"{roi_name}_tvb_scale.tif"
        tifffile.imwrite(str(tvb_path), pixel_maps['tvb_scale'].astype(np.float32))
        print(f"TVB scale map saved: {tvb_path}")


def create_complete_output_package(
    summary,
    pixel_maps,
    output_dir,
    roi_name,
    n_exp,
    strategy,
    metadata=None,
    save_individual_components=True,
    tau_display_min=None,
    tau_display_max=None,
    intensity_display_min=None,
    intensity_display_max=None,
    target_shape=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating complete output package: {roi_name}")
    print(f"{'='*60}\n")
    
    # Save fit summary
    summary_path = output_dir / f"{roi_name}_fit_summary.txt"
    save_fit_summary_txt(summary, summary_path, n_exp, strategy, metadata)
    
    # Save pixel maps if available
    if pixel_maps is not None:
        # Weighted averages
        save_weighted_tau_images(
            pixel_maps,
            output_dir,
            roi_name,
            n_exp,
            save_intensity=True,
            save_amplitude=True,
            tau_display_min=tau_display_min,
            tau_display_max=tau_display_max,
            intensity_display_min=intensity_display_min,
            intensity_display_max=intensity_display_max,
        )
        
        # Individual components
        if save_individual_components:
            save_individual_tau_maps(pixel_maps, output_dir, roi_name, n_exp)
    
    print(f"\n{'='*60}")
    print(f"Output package complete: {output_dir}")
    print(f"{'='*60}\n")
