from pathlib import Path

import numpy as np
import pytest

from flimkit import plugins
from flimkit.plugins import loader

FROZEN = Path(__file__).parent / 'fixtures' / 'plugin_api_v1' / 'frozen_v1_plugin.py'


@pytest.fixture
def frozen():
    loader.reset()
    plugins.ensure_loaded()
    result = plugins.load_path(str(FROZEN))
    assert result.ok == True, result.error
    yield result
    loader.reset()
    plugins.ensure_loaded()


def test_every_v1_hook_still_registers(frozen):
    assert frozen.n_registered == 5


def test_v1_tool_contract(frozen):
    found = plugins.get_tool('frozen_v1_tool')
    assert found.label == 'Frozen v1 Tool...'
    assert found.menu_path == ('Tools',)
    assert found.order == 910
    assert found.callback('the app') == 'the app'
    nested = plugins.get_tool('frozen_v1_nested')
    assert nested.menu_path == ('Tools', 'Frozen v1')


def test_v1_format_contract(frozen):
    from flimkit.formats import FLIMFile, detect_format, file_modality
    assert detect_format('a.frozenv1') == 'frozen_v1_format'
    assert file_modality('a.frozenv1') == 'time'
    found = plugins.get_format('frozen_v1_format')
    assert found.label == 'Frozen v1 Format'
    assert found.reader().__name__ == 'FrozenV1Reader'
    assert FLIMFile('a.frozenv1').path == 'a.frozenv1'


def test_v1_sniffer_contract(frozen, tmp_path):
    from flimkit.formats import detect_format
    target = tmp_path / 'thing.nothingknown'
    target.write_bytes(b'FROZENV1! and the rest')
    assert detect_format(str(target)) == 'frozen_v1_format'


def test_v1_phasor_filter_contract(frozen):
    from flimkit.phasor.filters import phasor_filter, phasor_filter_methods
    assert 'frozen_v1_filter' in phasor_filter_methods()
    real, imag = phasor_filter(np.ones((2, 2)), np.ones((2, 2)),
                               'frozen_v1_filter', sigma=4.0, size=7)
    assert real.max() == 4.0
    assert imag.max() == 4.0


def test_api_version_is_still_one(frozen):
    assert plugins.API_VERSION == 1
