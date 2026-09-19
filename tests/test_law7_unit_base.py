import inspect

import NeSST as nst
import NeSST.cross_sections as xs
import numpy as np
import pytest
from NeSST.constants import mat_dict


@pytest.fixture(scope="module")
def law7():
    """D(n,2n) is stored as an ENDF LAW7 tabulated double differential cross section"""
    mat = mat_dict["D"]
    ddx = mat.n2n_ddx
    assert type(ddx).__name__ == "doubledifferentialcrosssection_data"
    # the transform is opt in, so switch it on for these tests
    ddx.unit_base_N = 4096
    ddx.unit_base = True
    return ddx


def _grids():
    Ein = np.linspace(11.0e6, 16.0e6, 120)
    mu = np.linspace(-1.0, 1.0, 40)
    Eout = np.linspace(0.5e6, 16.0e6, 200)
    return Ein, mu, Eout


def _rgrid(ddx, Ein, mu, Eout):
    ddx.regular_grid(Ein, mu, Eout)
    return np.asarray(ddx.rgrid)


def _both_paths(law7):
    Ein, mu, Eout = _grids()
    law7.unit_base = True
    approx = _rgrid(law7, Ein, mu, Eout)
    law7.unit_base = False
    exact = _rgrid(law7, Ein, mu, Eout)
    law7.unit_base = True
    return approx, exact


def test_unit_base_is_off_by_default():
    """The exact evaluation is the default; the transform must be asked for"""
    signature = inspect.signature(xs.doubledifferentialcrosssection_data.__init__)
    assert signature.parameters["unit_base"].default is False
    assert signature.parameters["unit_base_N"].default is None


def test_enabling_without_a_resolution_raises(law7):
    with pytest.raises(ValueError, match="unit_base_N"):
        law7.build_unit_base_table(None)


def test_unit_base_table_is_built(law7):
    assert law7.unit_base
    assert law7.table.g.shape == (law7.NEin_ddx, max(law7.Ncos_ddx), law7.unit_base_N)


def test_unit_base_resampling_conserves_each_table(law7):
    """The unit base transform must not change the integral of any tabulated distribution"""
    u = np.linspace(0.0, 1.0, law7.unit_base_N)
    g = np.asarray(law7.table.g)
    for i in range(law7.NEin_ddx):
        for j in range(law7.Ncos_ddx[i]):
            x = np.asarray(law7.Eout_ddx[(i, j)])
            y = np.asarray(law7.f_ddx[(i, j)])
            Emax = law7.Emax_ddx[(i, j)]
            exact = np.trapezoid(y, x) if (Emax > 0.0 and x.size > 1) else 0.0
            if exact <= 0.0:
                continue
            assert np.trapezoid(g[i, j], u) * Emax == pytest.approx(exact, rel=1e-12)


def test_unit_base_toggle_is_not_cached(law7):
    """Switching the evaluation path must actually change the result"""
    approx, exact = _both_paths(law7)
    assert not np.array_equal(approx, exact)


def test_unit_base_matches_exact_ddx_grid(law7):
    approx, exact = _both_paths(law7)
    assert np.linalg.norm(approx - exact) / np.linalg.norm(exact) < 1e-3


def test_unit_base_matches_exact_spectrum(law7):
    Ein, mu, _ = _grids()
    approx, exact = _both_paths(law7)
    I_E = nst.QBrysk(Ein, *nst.DTprimspecmoments(5.0e3)[::2])

    def spectrum(rgrid):
        return np.trapezoid(I_E[:, None] * np.trapezoid(rgrid, mu, axis=1), Ein, axis=0)

    s_approx, s_exact = spectrum(approx), spectrum(exact)
    assert np.linalg.norm(s_approx - s_exact) / np.linalg.norm(s_exact) < 1e-4
    assert np.allclose(s_approx, s_exact, rtol=0.0, atol=1e-4 * np.abs(s_exact).max())


def test_unit_base_conserves_total_yield(law7):
    Ein, mu, Eout = _grids()
    approx, exact = _both_paths(law7)
    I_E = nst.QBrysk(Ein, *nst.DTprimspecmoments(5.0e3)[::2])

    def yield_(rgrid):
        return np.trapezoid(np.trapezoid(I_E[:, None] * np.trapezoid(rgrid, mu, axis=1), Ein, axis=0), Eout)

    assert yield_(approx) == pytest.approx(yield_(exact), rel=1e-5)


@pytest.mark.parametrize("Nu", [512, 2048])
def test_unit_base_error_falls_with_resolution(law7, Nu):
    """Refining the unit base grid must not make the approximation worse"""
    Ein, mu, Eout = _grids()
    law7.unit_base = False
    exact = _rgrid(law7, Ein, mu, Eout)
    law7.unit_base = True

    default_N = law7.unit_base_N
    law7.unit_base_N = Nu
    err_coarse = np.linalg.norm(_rgrid(law7, Ein, mu, Eout) - exact)
    law7.unit_base_N = 4 * Nu
    err_fine = np.linalg.norm(_rgrid(law7, Ein, mu, Eout) - exact)
    law7.unit_base_N = default_N

    assert err_fine < err_coarse
