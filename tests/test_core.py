import NeSST as nst
import numpy as np
import pytest
from NeSST.constants import LazyMaterialDict


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
    _, _, DTvar_hot = nst.DTprimspecmoments(Tion=10.0e3)  # units eV

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
    assert np.isclose((100 / DDmean) * DDstddev, 3, atol=0.3)
    # Check variance is standard deviation squared
    assert np.isclose(DDvar, DDstddev**2)


def test_DTprimspecmoments_variance_relative_size():
    # checks the relative magnitude of the var

    DTmean, DTstddev, DTvar = nst.DTprimspecmoments(Tion=5.0e3)  # units eV

    # Check that the standard deviation is about 3% of the mean value
    assert np.isclose((100 / DTmean) * DTstddev, 1, atol=0.3)
    # Check variance is standard deviation squared
    assert np.isclose(DTvar, DTstddev**2)


def test_mat_dict_is_lazy():
    # mat_dict should be a LazyMaterialDict (not loaded eagerly at import)
    assert isinstance(nst.mat_dict, LazyMaterialDict)


def test_mat_dict_lazy_loads_on_access():
    # mat_dict should lazily load material data on first access and then cache it
    # Be9 is accessed last among default materials, so test with it
    label = "Be9"
    # Ensure the label is accessible (loads lazily if not already cached)
    mat = nst.mat_dict[label]
    assert label in nst.mat_dict
    # Subsequent access returns the same cached object
    assert mat is nst.mat_dict[label]


def test_mat_dict_raises_for_unknown_label():
    # Accessing an unknown material label should raise KeyError
    with pytest.raises(KeyError):
        nst.mat_dict["unknown_material_xyz"]


###############################
# Bin averaged primary shapes #
###############################


@pytest.mark.parametrize("shape", [nst.QBrysk, nst.QBallabio])
def test_primary_bin_average_is_opt_in(shape):
    """The default must still sample the shape at the bin centres"""
    mean, _, var = nst.DTprimspecmoments(Tion=5.0e3)
    Ein = np.linspace(12.0e6, 16.0e6, 60)

    np.testing.assert_array_equal(shape(Ein, mean, var), shape(Ein, mean, var, bin_average=False))


@pytest.mark.parametrize("shape", [nst.QBrysk, nst.QBallabio])
def test_primary_bin_average_carries_the_same_yield_on_any_grid(shape):
    """A fusion primary is narrow, so sampling it at the bin centres loses area
    on a coarse grid.  Integrating over the bin must not."""
    from NeSST.core import Ecentres_to_edges

    mean, _, var = nst.DTprimspecmoments(Tion=5.0e3)

    yields = []
    for NE in (20, 40, 100, 1000):
        Ein = np.linspace(12.0e6, 16.0e6, NE)
        widths = np.asarray(Ecentres_to_edges(Ein)[1])
        yields.append(np.sum(shape(Ein, mean, var, bin_average=True) * widths))

    # every grid agrees with the finest one
    np.testing.assert_allclose(yields, yields[-1], rtol=1e-12)


def test_brysk_bin_average_is_normalised():
    """The Brysk shape is a normalised Gaussian, so its bin integral sums to one"""
    from NeSST.core import Ecentres_to_edges

    mean, _, var = nst.DTprimspecmoments(Tion=5.0e3)
    Ein = np.linspace(12.0e6, 16.0e6, 25)
    widths = np.asarray(Ecentres_to_edges(Ein)[1])

    assert np.sum(nst.QBrysk(Ein, mean, var, bin_average=True) * widths) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("shape", [nst.QBrysk, nst.QBallabio])
def test_primary_bin_average_matches_quadrature(shape):
    """The closed forms are exact, to roundoff, wherever the shape is resolved.
    They lose precision only in the deep tail, where the error function
    saturates and the bin integral cancels."""
    from itertools import pairwise

    from NeSST.core import Ecentres_to_edges
    from scipy.integrate import quad

    mean, _, var = nst.DTprimspecmoments(Tion=5.0e3)
    Ein = np.linspace(12.0e6, 16.0e6, 40)
    edges = np.asarray(Ecentres_to_edges(Ein)[0])

    got = shape(Ein, mean, var, bin_average=True)
    want = np.array(
        [
            quad(lambda x: float(shape(np.array([x]), mean, var)[0]), lo, hi, limit=400, epsabs=0.0, epsrel=1e-13)[0]
            / (hi - lo)
            for lo, hi in pairwise(edges)
        ]
    )

    resolved = want > want.max() * 1e-3
    np.testing.assert_allclose(got[resolved], want[resolved], rtol=1e-12)


@pytest.mark.parametrize("shape", [nst.QBrysk, nst.QBallabio])
def test_primary_bin_average_tends_to_the_point_value(shape):
    """Refining the grid must close the gap between the two"""
    mean, _, var = nst.DTprimspecmoments(Tion=5.0e3)

    errors = []
    for NE in (200, 800):
        Ein = np.linspace(13.0e6, 15.0e6, NE)
        point = shape(Ein, mean, var)
        binned = shape(Ein, mean, var, bin_average=True)
        errors.append(np.abs(binned / point - 1).max())

    assert errors[1] < errors[0] / 10.0
