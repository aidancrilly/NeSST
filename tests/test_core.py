import pytest
import NeSST as nst


def test_DTprimspecmoments():
    # checks the mean and variance value of the neutron emitted by DT fusion

    DTmean, _ = nst.DTprimspecmoments(Tion=5.0)  # units KeV

    assert DTmean == pytest.approx(14.1, abs=0.1)  # units MeV


def test_DDprimspecmoments():
    # checks the mean and variance value of the neutron emitted by DT fusion

    DTmean, _ = nst.DDprimspecmoments(Tion=5.0)  # units KeV

    assert DTmean == pytest.approx(2.5, abs=0.1)  # units MeV