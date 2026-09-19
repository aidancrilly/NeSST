"""Coverage for the scattering matrix construction path.

These exercise interp_Tlcoeff, diffxsec_legendre_eval, the SDX table
evaluation, the assembly in init_station_elastic_scatter and the opt-in bin
averaged kernels.
"""

import jax.numpy as jnp
import NeSST as nst
import NeSST.collisions as col
import NeSST.cross_sections as xs
import NeSST.spectral_model as sm
import numpy as np
import pytest
from NeSST import core
from NeSST.constants import mat_dict
from NeSST.utils import energy_bin_edges, midpoint_subnodes
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


#############################
# Opt-in bin averaged kernels #
#############################


def _sigma_bar(mat, Ein, M=512):
    """Elastic cross section averaged over the incoming bins"""
    edges, _ = energy_bin_edges(jnp.asarray(Ein))
    nodes, _ = midpoint_subnodes(edges[:-1], edges[1:], M)
    return np.asarray(jnp.mean(mat.sigma(nodes.reshape(-1)).reshape(-1, M), axis=1))


def _sum_rule_error(mat, Ein, Eout, N):
    """int dE' dNdEdmu(E->E') must return the elastic cross section"""
    _, dEout = energy_bin_edges(jnp.asarray(Eout))
    kernel = sm.BinAveragedElasticScatterKernel(A=mat.A, dxs=mat.elastic_dxs, N=N)
    _, K = kernel(jnp.asarray(Ein), jnp.asarray(Eout))
    total = np.asarray(jnp.sum(K * dEout[:, None], axis=0))
    return np.abs(total / _sigma_bar(mat, Ein) - 1).max()


@pytest.fixture(scope="module")
def carbon():
    nst.initialise_material_data("C12")
    return mat_dict["C12"]


@pytest.mark.parametrize("label", MATERIALS)
@pytest.mark.parametrize("classical", [True, None])
def test_muc_is_linear_in_outgoing_energy(label, classical):
    """The bin limits come from inverting a linearisation of mu_c in Eout, which
    only holds because mu_c is linear in Eout at fixed Ein.  That is exact for
    the classical kinematics; this pins down that it also holds for the
    relativistic kinematics over the energies NeSST is used at."""
    nst.initialise_material_data(label)
    A = mat_dict[label].A
    Ein = 14.1e6
    try:
        col.classical_collisions = classical
        # Sample across the whole band, avoiding the Eout -> 0 singularity
        g = col.g(A, Ein, Ein, 1.0, -1.0, 0.0)
        band_lo = Ein - 2.0 / g
        Eout = jnp.linspace(band_lo + 0.02 * (Ein - band_lo), Ein, 33)
        muc = col.muc(A, Ein, Eout, 1.0, -1.0, 0.0)
        # Second differences vanish for an exactly linear function
        curvature = jnp.max(jnp.abs(jnp.diff(muc, n=2)))
        assert curvature < 1.0e-10
    finally:
        col.classical_collisions = None


@pytest.mark.parametrize("label", MATERIALS)
def test_linearised_slope_matches_the_slowing_down_kernel(label):
    """For stationary targets the slope of mu_c in Eout is the jacobian g, so
    the AD linearisation and col.g must agree"""
    nst.initialise_material_data(label)
    A = mat_dict[label].A
    Ev = jnp.array([1.0e6, 5.0e6, 14.1e6])

    lo, hi, mu_probe, slope = sm.affine_muc_band(lambda Eo: col.muc(A, Ev, Eo, 1.0, -1.0, 0.0), Ev)
    g = col.g(A, Ev, Ev, 1.0, -1.0, 0.0)

    # The relativistic form divides by p*^2, a difference of large numbers, so
    # forward scatter lands on 1 to roundoff rather than exactly
    np.testing.assert_allclose(np.asarray(mu_probe), 1.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(slope), np.asarray(g), rtol=1e-10)
    # and the band it implies is the textbook one
    np.testing.assert_allclose(np.asarray(hi), np.asarray(Ev), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(lo), np.asarray(Ev - 2.0 / g), rtol=1e-9)


def test_bin_average_is_opt_in():
    """The default path must be untouched by the bin averaging machinery"""
    Ein = np.linspace(13.0e6, 15.0e6, 40)
    Eout = np.linspace(1.0e6, 15.0e6, 80)

    mat = core.init_mat_scatter(Eout, Ein, "C12")
    default = np.asarray(mat.elastic_dNdEdmu)
    mat = core.init_mat_scatter(Eout, Ein, "C12", bin_average=False)
    explicit = np.asarray(mat.elastic_dNdEdmu)

    np.testing.assert_array_equal(default, explicit)
    _, pointwise = mat.elastic_kernel(jnp.asarray(Ein), jnp.asarray(Eout))
    np.testing.assert_array_equal(default, np.asarray(pointwise))


def test_elastic_sum_rule_holds(carbon):
    """The bin averaged kernel conserves the elastic cross section"""
    Ein = np.linspace(13.0e6, 14.5e6, 20)
    Eout = np.linspace(1.0e3, 15.0e6, 2000)

    assert _sum_rule_error(carbon, Ein, Eout, N=8) < 1.0e-4


def test_elastic_sum_rule_converges_in_N(carbon):
    """Midpoint sub-division converges at second order"""
    Ein = np.linspace(13.0e6, 14.5e6, 20)
    Eout = np.linspace(1.0e3, 15.0e6, 2000)

    errors = [_sum_rule_error(carbon, Ein, Eout, N) for N in (2, 4, 8)]
    assert errors[1] < errors[0] / 3.0
    assert errors[2] < errors[1] / 3.0


def test_elastic_sum_rule_is_flat_in_outgoing_resolution(carbon):
    """The band edges are integrated exactly, so coarsening Eout must not cost
    cross section the way point sampling the kinematic limits does"""
    Ein = np.linspace(13.0e6, 14.5e6, 20)
    sig_c = np.asarray(carbon.sigma(jnp.asarray(Ein)))

    for NEout in (100, 1000):
        Eout = np.linspace(1.0e3, 15.0e6, NEout)
        _, K = carbon.elastic_kernel(jnp.asarray(Ein), jnp.asarray(Eout))
        pointwise = np.abs(np.asarray(jnp.trapezoid(K, jnp.asarray(Eout), axis=0)) / sig_c - 1).max()
        assert _sum_rule_error(carbon, Ein, Eout, N=4) < pointwise / 10.0


def test_inelastic_kernel_is_finite_below_threshold(carbon):
    """Every C12 level, including ones shut at these energies, stays finite"""
    Ein = np.linspace(1.0e5, 20.0e6, 60)
    Eout = np.linspace(1.0e3, 20.0e6, 300)

    for i_inelastic in range(carbon.n_inelastic):
        kernel = sm.BinAveragedInelasticScatterKernel(
            A=carbon.A,
            Q=carbon.inelasticQ[i_inelastic],
            dxs=carbon.inelastic_kernel[i_inelastic].dxs,
            N=2,
        )
        mu0, dNdEdmu = kernel(jnp.asarray(Ein), jnp.asarray(Eout))
        assert jnp.all(jnp.isfinite(dNdEdmu))
        assert jnp.all(jnp.isfinite(mu0))
        assert jnp.all(dNdEdmu >= 0.0)
        assert jnp.max(jnp.abs(mu0)) <= 1.0


def test_elastic_kernel_cosines_stay_physical(carbon):
    """rhoL_func is handed mu0, so it may never leave [-1, 1]"""
    Ein = np.linspace(1.0e5, 20.0e6, 60)
    Eout = np.linspace(1.0e3, 20.0e6, 300)

    kernel = sm.BinAveragedElasticScatterKernel(A=carbon.A, dxs=carbon.elastic_dxs, N=2)
    mu0, dNdEdmu = kernel(jnp.asarray(Ein), jnp.asarray(Eout))

    assert jnp.all(jnp.isfinite(dNdEdmu))
    assert jnp.max(jnp.abs(mu0)) <= 1.0


def test_energy_bin_edges_stay_non_negative():
    """A first centre within half a bin of zero must not mirror below zero"""
    Ein = np.linspace(1.0e5, 20.0e6, 60)
    edges, widths = energy_bin_edges(jnp.asarray(Ein))

    assert jnp.all(edges >= 0.0)
    assert jnp.all(widths > 0.0)
    # A grid clear of zero is untouched by the clamp
    clear = np.linspace(10.0e6, 15.0e6, 20)
    edges, _ = energy_bin_edges(jnp.asarray(clear))
    assert edges[0] == pytest.approx(clear[0] - 0.5 * (clear[1] - clear[0]))


def test_ion_kinematic_bin_average_shape_and_normalisation():
    """The bin averaged ion kinematic matrix keeps the point sampled contract"""
    nst.initialise_material_data("T")
    mat = mat_dict["T"]
    Ein = np.linspace(13.0e6, 14.5e6, 12)
    Eout = np.linspace(1.0e3, 15.0e6, 400)
    varr = np.array([-4.0e5, 0.0, 4.0e5])

    point_M, point_mu = mat.ion_kinematic_kernel(jnp.asarray(Eout), jnp.asarray(varr), jnp.asarray(Ein))
    kernel = sm.BinAveragedIonKinematicScatterKernel(A=mat.A, dxs=mat.elastic_dxs, N=1)
    M, mu = kernel(jnp.asarray(Eout), jnp.asarray(varr), jnp.asarray(Ein))

    assert M.shape == point_M.shape
    assert mu.shape == point_mu.shape
    assert jnp.all(jnp.isfinite(M))
    assert jnp.max(jnp.abs(mu)) <= 1.0

    # The stationary column must still integrate to the elastic cross section
    _, dEout = energy_bin_edges(jnp.asarray(Eout))
    total = np.asarray(jnp.sum(M[:, 1, :] * dEout[:, None], axis=0))
    assert np.abs(total / _sigma_bar(mat, Ein) - 1).max() < 1.0e-3
