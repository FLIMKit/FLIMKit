from flimkit.plugins import tool

FLIMKIT_PLUGIN_API = 1


@tool(id='irf_builder', label='Machine IRF Builder', menu='Tools', order=10)
def open_irf_builder(app):
    app._menu_irf_builder()


@tool(id='synth_generator', label='Generate Synthetic PTU...', menu='Tools', order=20)
def open_synth_generator(app):
    app._menu_synth_generator()


@tool(id='batch_tiled', label='Multi-Tile ROI Fit', menu='Tools/Batch Processing', order=110)
def open_batch_tiled(app):
    app._menu_batch_processing('tiled')


@tool(id='batch_fov', label='Single FOV Fit', menu='Tools/Batch Processing', order=120)
def open_batch_fov(app):
    app._menu_batch_processing('fov')


@tool(id='batch_timelapse', label='Timelapse Fit', menu='Tools/Batch Processing', order=130)
def open_batch_timelapse(app):
    app._menu_batch_processing('timelapse')
