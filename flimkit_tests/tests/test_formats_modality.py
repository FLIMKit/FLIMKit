from flimkit.formats import file_modality, detect_format

def test_modality_by_extension():
    assert file_modality('a.ptu') == 'time'
    assert file_modality('a.sdt') == 'time'
    assert file_modality('a.TAGTIME') == 'time'
    assert file_modality('a.tagdecay') == 'time'
    assert file_modality('a.ifli') == 'frequency'
    assert file_modality('a.ifi') == 'intensity'
    assert file_modality('a.photons') == 'time'
    assert file_modality('a.xyz') == 'unknown'

def test_modality_matches_detect():
    assert detect_format('a.ifli') == 'iss_fdflim'
    assert detect_format('a.ifi') == 'iss_image'
    assert detect_format('a.sdt') == 'bh_sdt'
    assert detect_format('a.photons') == 'ps'

BUILTIN_EXPECTED = {
    '.ptu': 'ptu',
    '.sdt': 'bh_sdt',
    '.tagtime': 'iss_tdflim',
    '.tagchannel': 'iss_tdflim',
    '.tagdecay': 'iss_tdflim',
    '.ifi': 'iss_image',
    '.ifli': 'iss_fdflim',
    '.photons': 'ps',
    '.phu': 'pq_phu',
    '.bin': 'pq_bin',
    '.b&h': 'simfcs_bh',
    '.bhz': 'simfcs_bhz',
    '.iss-tdflim': 'iss_vista_tdflim',
    '.tdflim': 'iss_vista_tdflim',
    '.ref': 'simfcs_referenced',
    '.r64': 'simfcs_referenced',
    '.xyz': 'unknown',
}

def test_every_builtin_extension_detects_the_same_id_with_plugins_loaded():
    from flimkit import plugins
    plugins.ensure_loaded()
    for ext, want in BUILTIN_EXPECTED.items():
        assert detect_format('a' + ext) == want

def test_plugin_format_does_not_displace_a_builtin_extension():
    from flimkit import plugins
    plugins.ensure_loaded()
    plugins.register_format('parity_probe', 'Parity Probe', exts=('.ptu',),
                            modality='time', reader='flimkit.formats.PTU.reader:PTUFile')
    try:
        assert detect_format('a.ptu') == 'ptu'
    finally:
        plugins.registry._formats.pop('parity_probe', None)
        plugins.registry._bump()
