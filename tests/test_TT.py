import NeSST as nst
import numpy as np
import pytest

# Load in TT spectrum
# Based on Appelbe, stationary emitter, temperature range between 1 and 20 keV
# https://www.sciencedirect.com/science/article/pii/S1574181816300295
# N.B. requires some unit conversion to uniform eV
TT_data = np.loadtxt(nst.sm.data_dir + "TT/TT_spec_temprange.txt")
TT_spec_E = TT_data[:, 0] * 1e6  # MeV to eV
TT_spec_T = np.linspace(1.0, 20.0, 40) * 1e3  # keV to eV
TT_spec_dNdE = TT_data[:, 1:] / 1e6  # 1/MeV to 1/eV
TT_2dinterp = nst.sm.interpolate_2d(
    TT_spec_E, TT_spec_T, TT_spec_dNdE, method="linear", bounds_error=False, fill_value=0.0
)


@pytest.mark.parametrize("Ti", TT_spec_T[1:])
def test_TT_spectrum(Ti):
    E = np.linspace(0.0, 12e6, 500)  # eV
    dNdE_interp = TT_2dinterp(E, Ti)
    dNdE_model = nst.dNdE_TT(E, Ti)

    assert np.allclose(dNdE_interp, dNdE_model, rtol=1e-3)
