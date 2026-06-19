import numpy as np
import pytest

import NeSST as nst


E_DT = np.linspace(12.5e6, 15.5e6, 60)
E_SCATTER = np.linspace(1.0e6, 16.0e6, 100)
TION = 5.0e3
RHOL = 0.5
FUEL_FRACTIONS = [0.6, 0.5, 0.4]


def _primary_weights(frac_D, frac_T):
    return (
        (frac_D / frac_T) * (nst.frac_T_default / nst.frac_D_default),
        (frac_T / frac_D) * (nst.frac_D_default / nst.frac_T_default),
    )


def _primary_scattering_source(fit, frac_D, frac_T):
    DD_weight, TT_weight = _primary_weights(frac_D, frac_T)
    return fit.I_DT(E_SCATTER) + DD_weight * fit.dNdE_DD + TT_weight * fit.dNdE_TT


def _primary_background(fit, frac_D, frac_T):
    DD_weight, TT_weight = _primary_weights(frac_D, frac_T)
    return DD_weight * fit.dNdE_DD + TT_weight * fit.dNdE_TT


@pytest.mark.parametrize("frac_D", FUEL_FRACTIONS)
def test_symmetric_fitting_model_matches_core_api(frac_D):
    frac_T = 1.0 - frac_D
    fit = nst.DT_fit_function(E_DT, E_SCATTER)
    fit.set_primary_Tion(TION)
    A_1S = nst.rhoR_2_A1s(RHOL, frac_D=frac_D, frac_T=frac_T)
    first_scatter, _ = nst.DT_sym_scatter_spec(
        _primary_scattering_source(fit, frac_D, frac_T), frac_D=frac_D, frac_T=frac_T
    )
    double_scatter, _ = nst.DT_sym_scatter_spec(first_scatter, frac_D=frac_D, frac_T=frac_T)

    fit.init_symmetric_model()
    single_result = fit.model(E_SCATTER, RHOL, 0.0, frac_T, frac_D, 1.0)
    np.testing.assert_allclose(
        single_result,
        A_1S * first_scatter + _primary_background(fit, frac_D, frac_T),
        rtol=1.0e-12,
        atol=1.0e-20,
    )

    fit.init_symmetric_model(include_double_scatter=True)
    double_result = fit.model(E_SCATTER, RHOL, 0.0, frac_T, frac_D, 1.0)
    np.testing.assert_allclose(
        double_result,
        A_1S * first_scatter + A_1S**2 * double_scatter + _primary_background(fit, frac_D, frac_T),
        rtol=1.0e-12,
        atol=1.0e-20,
    )


@pytest.mark.parametrize("frac_D", FUEL_FRACTIONS)
def test_modeone_fitting_model_matches_core_api(frac_D):
    frac_T = 1.0 - frac_D
    P1_arr = np.linspace(-0.5, 0.5, 5)
    P1 = P1_arr[-1]
    rhoL_func = lambda mu: 1.0 + P1 * mu
    fit = nst.DT_fit_function(E_DT, E_SCATTER)
    fit.set_primary_Tion(TION)
    A_1S = nst.rhoR_2_A1s(RHOL, frac_D=frac_D, frac_T=frac_T)
    first_scatter, _ = nst.DT_asym_scatter_spec(
        _primary_scattering_source(fit, frac_D, frac_T),
        rhoL_func,
        frac_D=frac_D,
        frac_T=frac_T,
    )
    double_scatter, _ = nst.DT_sym_scatter_spec(first_scatter, frac_D=frac_D, frac_T=frac_T)

    fit.init_modeone_model(P1_arr)
    single_result = fit.model(E_SCATTER, RHOL, P1, 0.0, frac_T, frac_D, 1.0)
    np.testing.assert_allclose(
        single_result,
        A_1S * first_scatter + _primary_background(fit, frac_D, frac_T),
        rtol=1.0e-12,
        atol=1.0e-20,
    )

    fit.init_modeone_model(P1_arr, include_double_scatter=True)
    double_result = fit.model(E_SCATTER, RHOL, P1, 0.0, frac_T, frac_D, 1.0)
    np.testing.assert_allclose(
        double_result,
        A_1S * first_scatter + A_1S**2 * double_scatter + _primary_background(fit, frac_D, frac_T),
        rtol=1.0e-12,
        atol=1.0e-20,
    )
