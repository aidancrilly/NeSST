import pytest
import numpy as np
import NeSST as nst


def test_DTprimspecmoments_mean():
    # checks the mean value of the neutron emitted by DT fusion

    DTmean, _, _ = nst.DTprimspecmoments(Tion=5.0e3)  # units eV

    assert DTmean == pytest.approx(14.1e6, abs=0.1e6)  # units eV


def test_DDprimspecmoments_mean():
    # checks the mean value of the neutron emitted by DD fusion

    DDmean, _, _ = nst.DDprimspecmoments(Tion=5.0e3)  # units eV

    assert DDmean == pytest.approx(2.5e6, abs=0.1e6)  # units eV


def test_DDprimspecmoments_mean_with_tion():
    # checks the energy of the neutron increases with ion temperature

    DDmean_cold, _, _ = nst.DDprimspecmoments(Tion=5.0e3)  # units eV
    DDmean_hot, _, _ = nst.DDprimspecmoments(Tion=10.0e3)  # units eV

    assert DDmean_cold < DDmean_hot  # units eV


def test_DTprimspecmoments_mean_with_tion():
    # checks the energy of the neutron increases with ion temperature

    DTmean_cold, _, _ = nst.DTprimspecmoments(Tion=5.0e3)  # units eV
    DTmean_hot, _, _ = nst.DTprimspecmoments(Tion=10.0e3)  # units eV

    assert DTmean_cold < DTmean_hot  # units eV

def test_DTprimspecmoments_variance_with_tion():
    # checks the relative magnitude of the var

    _, _, DTvar_cold = nst.DTprimspecmoments(Tion=5.0e3)  # units eV
    _, _,DTvar_hot = nst.DTprimspecmoments(Tion=10.0e3)  # units eV

    assert DTvar_cold < DTvar_hot  # units eV**2

def test_DDprimspecmoments_variance_with_tion():
    # checks the relative magnitude of the var

    _, _, DDvar_cold = nst.DDprimspecmoments(Tion=5.0e3)  # units eV
    _, _, DDvar_hot = nst.DDprimspecmoments(Tion=10.0e3)  # units eV

    assert DDvar_cold < DDvar_hot  # units eV**2

def test_DDprimspecmoments_variance_relative_size():
    # checks the relative magnitude of the var

    DDmean, DDstddev, DDvar = nst.DDprimspecmoments(Tion=5.0e3)  # units eV

    # Check that the standard deviation is about 3% of the mean value
    assert np.isclose((100/DDmean)*DDstddev, 3, atol=0.3)
    # Check variance is standard deviation squared
    assert np.isclose(DDvar, DDstddev**2)

def test_DTprimspecmoments_variance_relative_size():
    # checks the relative magnitude of the var

    DTmean, DTstddev, DTvar = nst.DTprimspecmoments(Tion=5.0e3)  # units eV

    # Check that the standard deviation is about 3% of the mean value
    assert np.isclose((100/DTmean)*DTstddev, 1, atol=0.3)
    # Check variance is standard deviation squared
    assert np.isclose(DTvar, DTstddev**2)