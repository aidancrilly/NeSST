import pytest
import numpy as np
import NeSST as nst

def test_IRF_normalisation():
    """
    
    For uniform sensitivity, the transform to ntof space should be conservative

    """
    nToF_distance = 20.0 # m
    flat_sens = nst.time_of_flight.get_unity_sensitivity()
    tophat_10cmthickscintillator = nst.time_of_flight.get_transit_time_tophat_IRF(scintillator_thickness=0.1)
    decayingGaussian_10cmthickscintillator = nst.time_of_flight.get_transit_time_tophat_w_decayingGaussian_IRF(scintillator_thickness=0.1,gaussian_FWHM=2.5e-9,decay_time=2e-9)

    E_ntof = np.linspace(1.0e6,10.0e6,500)
    DDmean,_,DDvar = nst.DDprimspecmoments(Tion=5.0e3)
    dNdE = nst.QBrysk(E_ntof,DDmean,DDvar)

    test_20m_nToF_1 = nst.time_of_flight.nToF(nToF_distance,flat_sens,tophat_10cmthickscintillator)
    t_det,normt_det,signal = test_20m_nToF_1.get_signal(E_ntof,dNdE)

    integral_signal_1 = np.trapz(y=signal,x=normt_det)

    test_20m_nToF_2 = nst.time_of_flight.nToF(nToF_distance,flat_sens,decayingGaussian_10cmthickscintillator)
    t_det,normt_det,signal = test_20m_nToF_2.get_signal(E_ntof,dNdE)

    integral_signal_2 = np.trapz(y=signal,x=normt_det)

    assert np.isclose(integral_signal_1,1.0,rtol=1e-3)
    assert np.isclose(integral_signal_2,1.0,rtol=1e-3)
