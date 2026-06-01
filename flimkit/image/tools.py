import cv2
import numpy as np
from pathlib import Path
from ..PTU import reader as ptufile


def make_intensity_image(ptu_path, rotate_90_cw=True, save_image=False):
    ptu = ptufile.PTUFile(ptu_path, verbose=False)
    stack = ptu.raw_pixel_stack(channel=ptu.photon_channel)  # (Y, X, H)
    intensity = stack.sum(axis=-1)  # sum over histogram bins → (Y, X)
    if rotate_90_cw:
        intensity = np.rot90(intensity, k=-1)  # 90° clockwise
        print('  Rotated 90° clockwise.')
    if save_image:
        out_path = Path(ptu_path).stem + '_intensity.png'
        normed = cv2.normalize(intensity.astype(np.float32), None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(out_path, normed)
        print(f"  Saved intensity image: {out_path}")
    return intensity


def make_cell_mask(intensity_image, save_mask=False, path=None, name=None,
                   flow_threshold=0.5, cellprob_threshold=0.0, resize_to=224,
                   gpu=True):
    try:
        from cellpose import models as cp_models
    except ImportError:
        raise ImportError("cellpose is not installed - pip install 'cellpose>=3.0'")

    # Accept both file paths and numpy arrays
    if isinstance(intensity_image, (str, Path)):
        img = cv2.imread(str(intensity_image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {intensity_image}")
    elif isinstance(intensity_image, np.ndarray):
        img = intensity_image
    else:
        raise TypeError(f"Expected file path or ndarray, got {type(intensity_image)}")

    if img.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {img.shape}")

    orig_h, orig_w = img.shape

    # Normalise to uint8 via 1st-99th percentile clip, matching what cellpose.org
    # receives when a PNG is uploaded (see cellpose_segment.py for rationale).
    lo, hi = np.percentile(img, [1, 99])
    span = float(hi - lo) if hi != lo else 1.0
    img_u8 = (np.clip((img.astype(np.float64) - lo) / span, 0.0, 1.0) * 255).astype(np.uint8)

    # Resize to 224×224 (matching cellpose.org pre-processing).
    if resize_to is not None and (orig_h != resize_to or orig_w != resize_to):
        img_input = cv2.resize(img_u8, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    else:
        img_input = img_u8

    model = cp_models.CellposeModel(gpu=gpu)
    labels, _, _ = model.eval(
        img_input,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={'tile_norm_blocksize': 0},
    )
    labels = labels.astype(np.int32)

    # Scale label map back to original resolution.
    if resize_to is not None and (orig_h != resize_to or orig_w != resize_to):
        labels = cv2.resize(labels, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    mask_bool = labels > 0

    if save_mask and path is not None:
        out_path = str(Path(path).with_suffix('')) + '_cell_mask.png'
        cv2.imwrite(out_path, mask_bool.astype(np.uint8) * 255)
        print(f"  Saved cell mask: {out_path}")

    return mask_bool


def apply_intensity_threshold(intensity_image, threshold):
    return intensity_image >= threshold


def pick_intensity_threshold(intensity_image, initial=None):
    import matplotlib
    matplotlib.use('TkAgg')          # need an interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    img = intensity_image.astype(float)
    vmax = img.max()
    if initial is None:
        initial = max(1, int(vmax * 0.05))

    # State container (mutable so the nested functions can write to it)
    state = {'threshold': int(initial)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.22)

    # Show intensity image
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.set_title(f"Intensity image  -  threshold = {state['threshold']} photons")
    ax.set_axis_off()

    # Red overlay for excluded pixels
    overlay_rgba = np.zeros((*img.shape, 4), dtype=float)
    mask_below = img < state['threshold']
    overlay_rgba[mask_below] = [1, 0, 0, 0.35]
    overlay_im = ax.imshow(overlay_rgba, interpolation='nearest')

    # Pixel count annotation
    n_above = int((~mask_below).sum())
    n_total = img.size
    count_text = ax.text(
        0.01, 0.01,
        f"{n_above:,}/{n_total:,} pixels kept ({100 * n_above / n_total:.1f} %)",
        transform=ax.transAxes, fontsize=9,
        color='white', backgroundcolor=(0, 0, 0, 0.5),
        verticalalignment='bottom',
    )

    # Slider
    ax_slider = plt.axes([0.15, 0.08, 0.65, 0.03])
    slider = Slider(ax_slider, 'Min photons', 0, max(int(vmax), 1),
                    valinit=initial, valstep=1, valfmt='%d')

    def _update(val):
        thr = int(val)
        state['threshold'] = thr
        mask_below = img < thr
        rgba = np.zeros((*img.shape, 4), dtype=float)
        rgba[mask_below] = [1, 0, 0, 0.35]
        overlay_im.set_data(rgba)
        n_above = int((~mask_below).sum())
        ax.set_title(f"Intensity image  -  threshold = {thr} photons")
        count_text.set_text(
            f"{n_above:,}/{n_total:,} pixels kept ({100 * n_above / n_total:.1f} %)"
        )
        fig.canvas.draw_idle()
    slider.on_changed(_update)
    # Accept button
    ax_btn = plt.axes([0.40, 0.02, 0.20, 0.04])
    btn = Button(ax_btn, 'Accept', hovercolor='lightgreen')
    def _accept(event):
        plt.close(fig)
    btn.on_clicked(_accept)
    # Also close on Enter key
    def _on_key(event):
        if event.key in ('enter', 'return'):
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    plt.show()   # blocks until window is closed
    chosen = state['threshold']
    print(f"  Intensity threshold selected: {chosen} photons")
    return chosen