import pytest
import NeSST as nst


def test_DTprimspecmoments_mean():
    # checks the mean value of the neutron emitted by DT fusion

    DTmean, _ = nst.DTprimspecmoments(Tion=5.0)  # units KeV

    assert DTmean == pytest.approx(14.1, abs=0.1)  # units MeV


def test_DDprimspecmoments_mean():
    # checks the mean value of the neutron emitted by DD fusion

    DDmean, _ = nst.DDprimspecmoments(Tion=5.0)  # units KeV

    assert DDmean == pytest.approx(2.5, abs=0.1)  # units MeV


def test_DDprimspecmoments_mean_with_tion():
    # checks the energy of the neutron increases with ion temperature

    DDmean_cold, _ = nst.DDprimspecmoments(Tion=5.0)  # units KeV
    DDmean_hot, _ = nst.DDprimspecmoments(Tion=10.0)  # units KeV

    assert DDmean_cold < DDmean_hot  # units MeV


def test_DTprimspecmoments_mean_with_tion():
    # checks the energy of the neutron increases with ion temperature

    DTmean_cold, _ = nst.DTprimspecmoments(Tion=5.0)  # units KeV
    DTmean_hot, _ = nst.DTprimspecmoments(Tion=10.0)  # units KeV

    assert DTmean_cold < DTmean_hot  # units MeV
