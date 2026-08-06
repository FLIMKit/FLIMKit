import numpy as np
import pandas as pd
import pytest

from flimkit.utils.xlsx_tools import load_irf_export, load_xlsx


def test_loads_lasx_csv_irf_columns(tmp_path):
    export = tmp_path / 'irf.csv'
    export.write_text(
        'LAS X diagram export\n'
        'Time [ns],Decay [counts],Time [ns],IRF [counts]\n'
        '0.0,20,0.0,1\n'
        '0.1,15,0.1,8\n'
        '0.2,10,0.2,2\n',
        encoding='utf-8-sig',
    )

    result = load_irf_export(export)

    np.testing.assert_allclose(result['irf_t'], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(result['irf_c'], [1.0, 8.0, 2.0])


def test_ignores_preamble_rows_wider_than_lasx_header(tmp_path):
    export = tmp_path / 'irf.csv'
    export.write_text(
        'metadata,with,more,fields,than,the,table\n'
        'Time [ns],Decay [counts],Time [ns],IRF [counts]\n'
        '0.0,20,0.0,1\n'
        '0.1,15,0.1,8\n'
    )

    result = load_irf_export(export)

    np.testing.assert_allclose(result['irf_t'], [0.0, 0.1])
    np.testing.assert_allclose(result['irf_c'], [1.0, 8.0])


def test_loads_semicolon_delimited_lasx_csv(tmp_path):
    export = tmp_path / 'irf.csv'
    export.write_text(
        'Time [ns];Decay [counts];Time [ns];IRF [counts]\n'
        '0.0;20;0.0;1\n'
        '0.1;15;0.1;8\n'
    )

    result = load_irf_export(export)

    np.testing.assert_allclose(result['irf_t'], [0.0, 0.1])
    np.testing.assert_allclose(result['irf_c'], [1.0, 8.0])


def test_loads_decimal_comma_lasx_csv(tmp_path):
    export = tmp_path / 'irf.csv'
    export.write_text(
        'Time [ns];Decay [counts];Time [ns];IRF [counts]\n'
        '0,0;20;0,0;1\n'
        '0,1;15;0,1;8\n'
    )

    result = load_irf_export(export)

    np.testing.assert_allclose(result['irf_t'], [0.0, 0.1])
    np.testing.assert_allclose(result['irf_c'], [1.0, 8.0])


def test_csv_without_lasx_header_names_the_file(tmp_path):
    export = tmp_path / 'bad.csv'
    export.write_text('time,counts\n0.0,1\n')

    with pytest.raises(ValueError, match=r'bad\.csv.*Time \['):
        load_irf_export(export)


def test_rejects_unsupported_irf_export_extension(tmp_path):
    export = tmp_path / 'irf.json'
    export.write_text('Time [ns],IRF [counts]\n0.0,1\n')

    with pytest.raises(ValueError, match=r'Expected \.xlsx'):
        load_irf_export(export)

def test_loads_tab_delimited_lasx_csv(tmp_path):
    export = tmp_path / 'irf.csv'
    export.write_text(
        'atto488 Lifetime Decay HyD X 1\t\tatto488 IRF HyD X 1\t\t\n'
        'Time [ns]\tLifetime Decay HyD X 1 [Counts]\tTime [ns]\tIRF HyD X 1 [Counts]\t\n'
        '0.04848485\t786.00000000\t1.06666667\t0.00000000\t\n'
        '0.14545455\t425.00000000\t1.16363636\t0.00018287\t\n',
        encoding='utf-8-sig',
    )

    result = load_irf_export(export)

    np.testing.assert_allclose(result['irf_t'], [1.06666667, 1.16363636])
    np.testing.assert_allclose(result['irf_c'], [0.0, 0.00018287])
    np.testing.assert_allclose(result['decay_c'], [786.0, 425.0])

def test_loads_ascii_export_extension(tmp_path):
    export = tmp_path / 'irf.dat'
    export.write_text('Time [ns],IRF [counts]\n0.0,1\n0.1,8\n')

    result = load_irf_export(export)

    np.testing.assert_allclose(result['irf_c'], [1.0, 8.0])


def test_legacy_xlsx_loader_remains_compatible(tmp_path):
    export = tmp_path / 'irf.xlsx'
    pd.DataFrame(
        [[0.0, 20, 0.0, 1], [0.1, 15, 0.1, 8]],
        columns=['Time [ns]', 'Decay [counts]', 'Time [ns]', 'IRF [counts]'],
    ).to_excel(export, index=False)

    generic = load_irf_export(export)
    legacy = load_xlsx(export)

    np.testing.assert_allclose(generic['irf_t'], legacy['irf_t'])
    np.testing.assert_allclose(generic['irf_c'], legacy['irf_c'])
