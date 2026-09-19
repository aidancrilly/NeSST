"""Coverage for the elastic scattering matrix construction path.

These exercise interp_Tlcoeff, diffxsec_legendre_eval, the SDX table
evaluation and the assembly in init_station_elastic_scatter, none of which
were reached by the rest of the suite.
"""

import NeSST as nst
import NeSST.cross_sections as xs
import numpy as np
import pytest
from NeSST.constants import mat_dict
from numpy.polynomial.legendre import legval as np_legval
from scipy.interpolate import griddata

MATERIALS = ["D", "T", "C12"]


@pytest.fixture(scope="module")
def energy_grids():
    Ein = np.linspace(11.0e6, 16.0e6, 40)
    Eout = np.linspace(0.05e6, 16.0e6, 2000)
    return Ein, Eout


def test_legendre_eval_matches_numpy():
    """diffxsec_legendre_eval must agree with numpy's legval, tensor=False"""
    rng = np.random.default_rng(0)
    mu = np.linspace(-1.2, 1.2, 61)
    coeff = rng.normal(size=(mu.size, 5))
    sig = np.linspace(1.0, 2.0, mu.size)

    got = np.asarray(xs.diffxsec_legendre_eval(sig, mu, coeff))
    expected = sig * np_legval(mu, coeff.T, tensor=False)
    expected = np.where(np.abs(mu) > 1.0, 0.0, expected)

    assert np.allclose(got, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("label", MATERIALS)
def test_differential_cross_section_normalisation(label):
    """Integrating the CoM differential cross section over mu returns sigma"""
    mat = mat_dict[label]
    Ein = np.linspace(11.0e6, 15.0e6, 5)
    mu = np.linspace(-1.0, 1.0, 20001)

    dsdO = np.asarray(xs.f_dsdO(Ein, np.broadcast_to(mu[:, None], (mu.size, Ein.size)), mat.elastic_dxs))
    integral = np.trapezoid(dsdO, mu, axis=0)

    assert np.allclose(integral, np.asarray(mat.sigma(Ein)), rtol=1e-6)


@pytest.mark.parametrize("label", MATERIALS)
def test_elastic_scatter_matrix_is_well_formed(label, energy_grids):
    Ein, Eout = energy_grids
    mat = nst.init_mat_scatter(Eout, Ein, label)
    M = np.asarray(mat.elastic_dNdEdmu)

    assert M.shape == (Eout.size, Ein.size)
    assert np.all(np.isfinite(M))
    assert np.all(M >= 0.0)
    assert M.max() > 0.0

    # the lab cosine is only physical inside the kinematically allowed band,
    # which is exactly where the matrix carries weight
    mu0 = np.asarray(mat.elastic_mu0)
    assert np.all(np.abs(mu0[M > 0.0]) <= 1.0 + 1e-12)


@pytest.mark.parametrize("label", MATERIALS)
def test_elastic_scatter_matrix_conserves_cross_section(label, energy_grids):
    """Integrating the scatter matrix over outgoing energy returns sigma.

    This covers the whole assembly at once: the legendre coefficients, the
    centre of mass cosine and the energy to cosine jacobian.
    """
    Ein, Eout = energy_grids
    mat = nst.init_mat_scatter(Eout, Ein, label)

    integral = np.trapezoid(np.asarray(mat.elastic_dNdEdmu), Eout, axis=0)

    assert np.allclose(integral, np.asarray(mat.sigma(Ein)), rtol=1e-2)


def test_sdx_table_matches_griddata():
    """The Delaunay evaluation must reproduce scipy's griddata.

    No shipped material uses the tabulated SDX path, so it is exercised here
    against a synthetic ragged table in the ENDF LTT=2 layout.
    """
    rng = np.random.default_rng(3)
    Ein = np.linspace(1.0e6, 20.0e6, 25)
    points, values = [], []
    for E in Ein:
        n = rng.integers(9, 21)
        mu = np.sort(np.concatenate([[-1.0, 1.0], rng.uniform(-1.0, 1.0, n - 2)]))
        points.append(np.column_stack([np.full(mu.size, E), mu]))
        values.append(0.5 + 0.3 * np.cos(3 * mu) + 1e-7 * E * mu**2)
    points = np.concatenate(points)
    values = np.concatenate(values)
    table = xs.NeSST_SDX(Ein=Ein, points=points, values=values)

    Eq, muq = np.meshgrid(np.linspace(1.5e6, 19.0e6, 40), np.linspace(-1.3, 1.3, 37), indexing="ij")
    sig = np.linspace(1.0, 2.0, Eq.shape[0])[:, None]

    expected = sig * griddata(points, values, np.column_stack((Eq.ravel(), muq.ravel())), rescale=True).reshape(
        muq.shape
    )
    expected = np.nan_to_num(np.where(np.abs(muq) > 1.0, 0.0, expected), nan=0.0)

    got = np.asarray(xs.diffxsec_table_eval(sig, muq, Eq, table))

    assert np.allclose(got, expected, rtol=0.0, atol=1e-12)
