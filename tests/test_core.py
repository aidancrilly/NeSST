import pytest
import NeSST as nst


def test_DTprimspecmoments_mean():
    # checks the mean value of the neutron emitted by DT fusion

    DTmean, _ = nst.DTprimspecmoments(Tion=5.0e3)  # units eV

    assert DTmean == pytest.approx(14.1e6, abs=0.1e6)  # units eV


def test_DDprimspecmoments_mean():
    # checks the mean value of the neutron emitted by DD fusion

    DDmean, _ = nst.DDprimspecmoments(Tion=5.0e3)  # units eV

    assert DDmean == pytest.approx(2.5e6, abs=0.1e6)  # units eV


def test_DDprimspecmoments_mean_with_tion():
    # checks the energy of the neutron increases with ion temperature

    DDmean_cold, _ = nst.DDprimspecmoments(Tion=5.0e3)  # units eV
    DDmean_hot, _ = nst.DDprimspecmoments(Tion=10.0e3)  # units eV

    assert DDmean_cold < DDmean_hot  # units eV


def test_DTprimspecmoments_mean_with_tion():
    # checks the energy of the neutron increases with ion temperature

    DTmean_cold, _ = nst.DTprimspecmoments(Tion=5.0e3)  # units eV
    DTmean_hot, _ = nst.DTprimspecmoments(Tion=10.0e3)  # units eV

    assert DTmean_cold < DTmean_hot  # units eV
