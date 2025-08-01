import NeSST as nst
import numpy as np
import pytest_cases

@pytest_cases.fixture(params=['delta','decaying_gaussian','gated_decaying_gaussian_kernel','double_decay_gaussian_kernel','t_gaussian_kernel'])
def kernel_fn(request):
    if request.param == 'delta':
        kernel = lambda t: np.array([1.0])
    elif request.param == 'decaying_gaussian':
        kernel = nst.time_of_flight.decaying_gaussian_kernel(FWHM=2.5e-9,tau=2e-9)
    elif request.param == 'gated_decaying_gaussian_kernel':
        kernel = nst.time_of_flight.gated_decaying_gaussian_kernel(sig=1.0e-9,tau=2e-9,shift_t=1.0e-9, sig_turnon=0.5e-9)
    elif request.param == 'double_decay_gaussian_kernel':
        kernel = nst.time_of_flight.double_decay_gaussian_kernel(FWHM=2.5e-9,taus=[2e-9,10e-9],frac=0.5)
    elif request.param == 't_gaussian_kernel':
        kernel = nst.time_of_flight.t_gaussian_kernel(FWHM=2.5e-9,peak_pos=2e-9)

    return kernel




def test_IRF_normalisation(kernel_fn):
    """

    For uniform sensitivity, the transform to ntof space should be conservative

    """
    nToF_distance = 20.0  # m
    flat_sens = nst.time_of_flight.get_unity_sensitivity()
    total_IRF = nst.time_of_flight.make_transit_time_IRF(thickness = 0.1, kernel_fn = kernel_fn)

    E_ntof = np.linspace(1.0e6, 10.0e6, 500)
    DDmean, _, DDvar = nst.DDprimspecmoments(Tion=5.0e3)
    dNdE = nst.QBrysk(E_ntof, DDmean, DDvar)

    test_20m_nToF = nst.time_of_flight.nToF(nToF_distance, flat_sens, total_IRF)
    t_det, normt_det, signal = test_20m_nToF.get_signal(E_ntof, dNdE)

    integral_signal = np.trapezoid(y=signal, x=normt_det)

    assert np.isclose(integral_signal, 1.0, rtol=1e-3)

