import pytest
import numpy as np
import NeSST as nst

def test_endf():
    ENDF_data = nst.sm.retrieve_ENDF_data('H2.json')

    assert ENDF_data['interactions'].total
    assert ENDF_data['interactions'].elastic
    assert ENDF_data['interactions'].n2n

    assert ENDF_data['elastic_dxsec']['legendre']

    assert np.isclose(ENDF_data['n2n_dxsec']['Q_react'],-2.224640e+6)