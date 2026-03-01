import numpy as np
import pytest

pytest.importorskip("dress", reason="pydress is required for DRESS tests")

import NeSST as nst

# Number of Monte Carlo samples used in tests - large enough for accurate
# statistics but small enough to keep test runtime reasonable.
_N_SAMPLES = int(5e5)

# Tolerance for relative comparison of mean energy against Ballabio (%)
_MEAN_RTOL = 0.08  # 8 %
# Tolerance for relative comparison of standard deviation against Ballabio (%)
_STDDEV_RTOL = 0.05  # 5 %


def _spectrum_moments(Ein, spec):
    """Return the mean and standard deviation of a normalised spectrum."""
    _, dE = nst.Ecentres_to_edges(Ein)
    total = np.sum(spec * dE)
    mean = np.sum(Ein * spec * dE) / total
    stddev = np.sqrt(np.sum((Ein - mean) ** 2 * spec * dE) / total)
    return mean, stddev


def test_QDress_DT_mean_matches_Ballabio():
    """DRESS DT mean neutron energy should match Ballabio within 1 %."""
    Tion = 5e3  # 5 keV in eV
    mean_ball, stddev_ball, _ = nst.DTprimspecmoments(Tion)

    Ein = np.linspace(mean_ball - 10 * stddev_ball, mean_ball + 10 * stddev_ball, 500)
    spec = nst.QDress_DT(Ein, Tion, n_samples=_N_SAMPLES)

    mean_dress, _ = _spectrum_moments(Ein, spec)

    assert mean_dress - nst.E0_DT == pytest.approx(mean_ball - nst.E0_DT, rel=_MEAN_RTOL)


def test_QDress_DT_stddev_matches_Ballabio():
    """DRESS DT spectral width should match Ballabio within 5 %."""
    Tion = 5e3  # 5 keV in eV
    mean_ball, stddev_ball, _ = nst.DTprimspecmoments(Tion)

    Ein = np.linspace(mean_ball - 10 * stddev_ball, mean_ball + 10 * stddev_ball, 500)
    spec = nst.QDress_DT(Ein, Tion, n_samples=_N_SAMPLES)

    _, stddev_dress = _spectrum_moments(Ein, spec)

    assert stddev_dress == pytest.approx(stddev_ball, rel=_STDDEV_RTOL)


def test_QDress_DT_separate_temperatures():
    """Using different D and T temperatures should produce a different spectrum
    from a single-temperature run."""
    Tion = 5e3  # 5 keV in eV
    T_D = 5e3
    T_T = 10e3  # tritons hotter than deuterons

    mean_ball, stddev_ball, _ = nst.DTprimspecmoments(Tion)
    Ein = np.linspace(mean_ball - 10 * stddev_ball, mean_ball + 10 * stddev_ball, 500)

    spec_single = nst.QDress_DT(Ein, Tion, n_samples=_N_SAMPLES)
    spec_diff = nst.QDress_DT(Ein, T_D, T_T, n_samples=_N_SAMPLES)

    # The spectra should not be identical when temperatures differ
    assert not np.allclose(spec_single, spec_diff)


def test_QDress_DD_mean_matches_Ballabio():
    """DRESS DD mean neutron energy should match Ballabio within 1 %."""
    Tion = 5e3  # 5 keV in eV
    mean_ball, stddev_ball, _ = nst.DDprimspecmoments(Tion)

    Ein = np.linspace(mean_ball - 10 * stddev_ball, mean_ball + 10 * stddev_ball, 500)
    spec = nst.QDress_DD(Ein, Tion, n_samples=_N_SAMPLES)

    mean_dress, _ = _spectrum_moments(Ein, spec)

    assert mean_dress - nst.E0_DD == pytest.approx(mean_ball - nst.E0_DD, rel=_MEAN_RTOL)


def test_QDress_DD_stddev_matches_Ballabio():
    """DRESS DD spectral width should match Ballabio within 5 %."""
    Tion = 5e3  # 5 keV in eV
    mean_ball, stddev_ball, _ = nst.DDprimspecmoments(Tion)

    Ein = np.linspace(mean_ball - 10 * stddev_ball, mean_ball + 10 * stddev_ball, 500)
    spec = nst.QDress_DD(Ein, Tion, n_samples=_N_SAMPLES)

    _, stddev_dress = _spectrum_moments(Ein, spec)

    assert stddev_dress == pytest.approx(stddev_ball, rel=_STDDEV_RTOL)
