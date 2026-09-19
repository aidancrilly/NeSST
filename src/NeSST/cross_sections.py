# Backend of spectral model
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array
from scipy.spatial import Delaunay

import NeSST.collisions as col
from NeSST.constants import *
from NeSST.utils import *

###############################
# Differential cross sections #
###############################


class NeSST_SDX(eqx.Module):
    Ein: npt.NDArray
    points: npt.NDArray
    values: npt.NDArray
    simplices: Array
    transform: Array
    offset: Array
    scale: Array

    def __init__(self, Ein, points, values):
        self.Ein = Ein
        self.points = points
        self.values = values
        points = np.asarray(points, dtype=np.float64)
        offset = points.min(axis=0)
        scale = np.ptp(points, axis=0)
        scale[scale == 0.0] = 1.0
        tri = Delaunay((points - offset) / scale)
        self.simplices = jnp.asarray(tri.simplices)
        self.transform = jnp.asarray(tri.transform)
        self.offset = jnp.asarray(offset)
        self.scale = jnp.asarray(scale)

    def __call__(self, E, mu):
        shape = jnp.shape(mu)
        xi = jnp.stack([jnp.ravel(jnp.broadcast_to(E, shape)), jnp.ravel(mu)], axis=-1)
        xi = (xi - self.offset) / self.scale
        values = jnp.asarray(self.values)

        def evaluate(x):
            delta = x[None, :] - self.transform[:, 2, :]
            bary = jnp.einsum("tij,tj->ti", self.transform[:, :2, :], delta)
            bary = jnp.concatenate([bary, 1.0 - bary.sum(axis=1, keepdims=True)], axis=1)
            inside = jnp.all(bary >= -1e-12, axis=1) & jnp.all(jnp.isfinite(bary), axis=1)
            t = jnp.argmax(inside)
            f = jnp.sum(bary[t] * values[self.simplices[t]])
            return jnp.where(inside[t], f, jnp.nan)

        return jax.lax.map(evaluate, xi, batch_size=64).reshape(shape)


@dataclass
class NeSST_DDX:
    NEin: int
    Ein: list
    Ncos: list
    cos: list
    NEout: dict
    Eout: dict
    f: dict
    Emax: dict


# Elastic single differential cross sections


def diffxsec_table_eval(sig, mu, E, table):
    ans = sig * table(E, mu)
    return jnp.where(jnp.abs(mu) > 1.0, 0.0, jnp.nan_to_num(ans, nan=0.0))


# Interpolate the legendre coefficients (a_l) of the differential cross section
# See https://t2.lanl.gov/nis/endf/intro20.html
def interp_Tlcoeff(legendre_dx_spline, E_vec):
    NTl = len(legendre_dx_spline)
    Tlcoeff = jnp.stack([spline(E_vec) for spline in legendre_dx_spline], axis=-1)
    return Tlcoeff, NTl


# Evaluate the differential cross section by combining legendre and cross section
# See https://t2.lanl.gov/nis/endf/intro20.html
@jax.jit
def diffxsec_legendre_eval(sig, mu, coeff):
    c = coeff.T
    if mu.ndim == 1:
        ans = sig * legval(mu, c)
    elif mu.ndim == 2:
        ans = sig * legval(mu, c[:, None, :])
    elif mu.ndim == 3:
        ans = sig * legval(mu, c[:, None, None, :])
    return jnp.where(jnp.abs(mu) > 1.0, 0.0, ans)


class DifferentialCrossSection(eqx.Module):
    """sigma(Ein) dsigma/dOmega(mu), from legendre coefficients or a tabulated SDX"""

    sigma: Interpolator1D
    legendre: tuple
    SDX: NeSST_SDX

    def __call__(self, Ein_vec, mu, E_table=None):
        sig = self.sigma(Ein_vec)
        if self.legendre is None:
            return diffxsec_table_eval(sig, mu, E_table, self.SDX)
        Tlcoeff, Nl = interp_Tlcoeff(self.legendre, Ein_vec)
        Tlcoeff_interp = 0.5 * (2 * jnp.arange(0, Nl) + 1) * Tlcoeff
        return diffxsec_legendre_eval(sig, mu, Tlcoeff_interp)


# CoM frame differential cross section wrapper fucntion
@eqx.filter_jit
def f_dsdO(Ein_vec, mu, dxs):
    return dxs(Ein_vec, mu)


# Differential cross section even larger wrapper function
@eqx.filter_jit
def dsigdOmega(A, Ein, Eout, Ein_vec, muin, muout, vf, dxs):
    mu_CoM = col.muc(A, Ein, Eout, muin, muout, vf)
    return dxs(Ein_vec, mu_CoM)


class LAW7Table(eqx.Module):
    """Padded ENDF LAW7 table evaluated with the unit base transform"""

    xsec: Interpolator1D
    Ein: Array
    cos: Array
    Ncos: Array
    Emax: Array
    Eout: Array
    f: Array
    NEin: int = eqx.field(static=True)

    def _f_interp(self, iE, ic, Eout):
        x = self.Eout[iE, ic]
        y = self.f[iE, ic]
        f = jnp.interp(Eout, x, y, left=0.0, right=0.0)
        return jnp.where((Eout < x[0]) | (Eout > x[-1]), 0.0, f)

    # Interpolate using Unit Base Transform
    def __call__(self, Ein, mu, Eout):
        Ein = jnp.asarray(Ein)
        mu = jnp.asarray(mu)
        Eout = jnp.asarray(Eout)

        # Find indices
        # Energies
        Eidx2 = jnp.clip(jnp.searchsorted(self.Ein, Ein, side="right"), 1, self.NEin - 1)
        Eidx2 = jnp.where(Ein == self.Ein[-1], self.NEin - 1, Eidx2)
        Eidx1 = Eidx2 - 1

        # Angles
        def cos_index(iE):
            Nc = self.Ncos[iE]
            c = self.cos[iE]
            idx2 = jnp.clip(jnp.searchsorted(c, mu, side="right"), 1, Nc - 1)
            idx2 = jnp.where(mu == +1.0, Nc - 1, idx2)
            idx2 = jnp.where(mu == -1.0, 1, idx2)
            return idx2, idx2 - 1

        Cidx12, Cidx11 = cos_index(Eidx1)
        Cidx22, Cidx21 = cos_index(Eidx2)

        # Find interpolation factors
        mu_x1 = (mu - self.cos[Eidx1, Cidx11]) / (self.cos[Eidx1, Cidx12] - self.cos[Eidx1, Cidx11])
        mu_x2 = (mu - self.cos[Eidx2, Cidx21]) / (self.cos[Eidx2, Cidx22] - self.cos[Eidx2, Cidx21])
        Ein_x = (Ein - self.Ein[Eidx1]) / (self.Ein[Eidx2] - self.Ein[Eidx1])

        x_112 = mu_x1
        x_111 = 1 - x_112
        x_222 = mu_x2
        x_221 = 1 - x_222

        x_2 = Ein_x
        x_1 = 1 - x_2

        # Unit base transform
        E_h11 = self.Emax[Eidx1, Cidx11]
        E_h12 = self.Emax[Eidx1, Cidx12]
        E_h21 = self.Emax[Eidx2, Cidx21]
        E_h22 = self.Emax[Eidx2, Cidx22]
        E_h1 = E_h11 + mu_x1 * (E_h12 - E_h11)
        E_h2 = E_h21 + mu_x2 * (E_h22 - E_h21)
        E_high = E_h1 + Ein_x * (E_h2 - E_h1)
        E_high_safe = jnp.where(E_high == 0.0, 1.0, E_high)

        J_111 = E_h11 / E_high_safe
        J_112 = E_h12 / E_high_safe
        J_221 = E_h21 / E_high_safe
        J_222 = E_h22 / E_high_safe

        # Find unit base transformed energy
        f_111 = self._f_interp(Eidx1, Cidx11, Eout * J_111) * J_111
        f_112 = self._f_interp(Eidx1, Cidx12, Eout * J_112) * J_112
        f_221 = self._f_interp(Eidx2, Cidx21, Eout * J_221) * J_221
        f_222 = self._f_interp(Eidx2, Cidx22, Eout * J_222) * J_222

        f_1 = x_111 * f_111 + x_112 * f_112
        f_2 = x_221 * f_221 + x_222 * f_222

        f_ddx = x_1 * f_1 + x_2 * f_2

        f_ddx = jnp.where(E_high == 0.0, 0.0, f_ddx)
        return jnp.where(Ein < self.Ein[0], 0.0, f_ddx)

    @eqx.filter_jit
    def regular_grid(self, Ein, mu, Eout):
        return jax.vmap(lambda E: 2.0 * self.xsec(E) * jax.vmap(lambda m: self(E, m, Eout))(mu))(Ein)


# Inelastic double differential cross sections
# Reads and interpolated data saved in the ENDF interpreted data format
class doubledifferentialcrosssection_data:
    def __init__(self, ENDF_LAW6_xsec_data, ENDF_LAW6_dxsec_data):
        self.xsec_interp = interpolate_1d(
            ENDF_LAW6_xsec_data["E"], ENDF_LAW6_xsec_data["sig"], method="linear", bounds_error=False, fill_value=0.0
        )

        DDX = ENDF_LAW6_dxsec_data["DDX"]
        self.NEin_ddx = DDX.NEin
        self.Ein_ddx = DDX.Ein
        self.Ncos_ddx = DDX.Ncos
        self.cos_ddx = DDX.cos
        self.NEout_ddx = DDX.NEout
        self.Eout_ddx = DDX.Eout
        self.f_ddx = DDX.f
        self.Emax_ddx = DDX.Emax

        # Pad the ragged (Ein, cos) table onto dense arrays for vectorised evaluation
        max_Ncos = max(self.Ncos_ddx)
        max_NEout = max(self.NEout_ddx.values())
        cos_pad = np.zeros((self.NEin_ddx, max_Ncos))
        Emax_pad = np.zeros((self.NEin_ddx, max_Ncos))
        Eout_pad = np.zeros((self.NEin_ddx, max_Ncos, max_NEout))
        f_pad = np.zeros((self.NEin_ddx, max_Ncos, max_NEout))
        for i in range(self.NEin_ddx):
            Nc = self.Ncos_ddx[i]
            cos_pad[i, :Nc] = self.cos_ddx[i]
            cos_pad[i, Nc:] = self.cos_ddx[i][-1]
            for j in range(Nc):
                NEo = self.NEout_ddx[(i, j)]
                Emax_pad[i, j] = self.Emax_ddx[(i, j)]
                Eout_pad[i, j, :NEo] = self.Eout_ddx[(i, j)]
                Eout_pad[i, j, NEo:] = self.Eout_ddx[(i, j)][-1]
                f_pad[i, j, :NEo] = self.f_ddx[(i, j)]

        self.table = LAW7Table(
            xsec=self.xsec_interp,
            Ein=jnp.asarray(self.Ein_ddx),
            cos=jnp.asarray(cos_pad),
            Ncos=jnp.asarray(self.Ncos_ddx),
            Emax=jnp.asarray(Emax_pad),
            Eout=jnp.asarray(Eout_pad),
            f=jnp.asarray(f_pad),
            NEin=self.NEin_ddx,
        )

    def interpolate(self, Ein, mu, Eout):
        return self.table(Ein, mu, Eout)

    def regular_grid(self, Ein, mu, Eout):
        self.rgrid_shape = (Ein.shape[0], mu.shape[0], Eout.shape[0])
        grid = self.table.regular_grid(jnp.asarray(Ein), jnp.asarray(mu), jnp.asarray(Eout))
        self.rgrid = grid.reshape(self.rgrid_shape)


class LAW6Kernel(eqx.Module):
    """Analytic ENDF LAW6 phase space double differential cross section"""

    xsec: Interpolator1D
    A_i: float
    A_e: float
    A_t: float
    A_p: float
    A_tot: float
    Q_react: float

    def __call__(self, Ein, mu, Eout):
        E_star = Ein * self.A_i * self.A_e / (self.A_t + self.A_i) ** 2
        E_a = self.A_t * Ein / (self.A_p + self.A_t) + self.Q_react
        E_max = (self.A_tot - 1.0) * E_a / self.A_tot
        C3 = 4.0 / (jnp.pi * E_max * E_max)
        square_bracket_term = E_max - (E_star + Eout - 2 * mu * jnp.sqrt(E_star * Eout))
        square_bracket_term = jnp.where(square_bracket_term < 0.0, 0.0, square_bracket_term)
        f_ddx = C3 * jnp.sqrt(Eout * square_bracket_term)
        return f_ddx

    @eqx.filter_jit
    def regular_grid(self, Ein, mu, Eout):
        Ei, Mm, Eo = jnp.meshgrid(Ein, mu, Eout, indexing="ij")
        return 2.0 * self.xsec(Ei) * self(Ei, Mm, Eo)


class doubledifferentialcrosssection_LAW6:
    def __init__(self, ENDF_LAW6_xsec_data, ENDF_LAW6_dxsec_data):
        self.A_i = ENDF_LAW6_dxsec_data["A_i"]
        self.A_e = ENDF_LAW6_dxsec_data["A_e"]
        self.A_t = ENDF_LAW6_dxsec_data["A_t"]
        self.A_p = ENDF_LAW6_dxsec_data["A_p"]
        self.A_tot = ENDF_LAW6_dxsec_data["A_tot"]
        self.Q_react = ENDF_LAW6_dxsec_data["Q_react"]
        self.xsec_interp = interpolate_1d(
            ENDF_LAW6_xsec_data["E"], ENDF_LAW6_xsec_data["sig"], method="linear", bounds_error=False, fill_value=0.0
        )
        self.kernel = LAW6Kernel(
            xsec=self.xsec_interp,
            A_i=self.A_i,
            A_e=self.A_e,
            A_t=self.A_t,
            A_p=self.A_p,
            A_tot=self.A_tot,
            Q_react=self.Q_react,
        )

    def ddx(self, Ein, mu, Eout):
        return self.kernel(Ein, mu, Eout)

    def regular_grid(self, Ein, mu, Eout):
        self.rgrid = self.kernel.regular_grid(jnp.asarray(Ein), jnp.asarray(mu), jnp.asarray(Eout))
