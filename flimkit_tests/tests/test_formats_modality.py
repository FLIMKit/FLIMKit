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
