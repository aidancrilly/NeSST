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


# Inelastic double differential cross sections
# Reads and interpolated data saved in the ENDF interpreted data format
class doubledifferentialcrosssection_data:
    def __init__(self, ENDF_LAW6_xsec_data, ENDF_LAW6_dxsec_data):
        self.xsec_interp = interpolate_1d(
            ENDF_LAW6_xsec_data["E"], ENDF_LAW6_xsec_data["sig"], method="linear", bounds_error=False, fill_value=0.0
        )

        self.NEin_ddx = ENDF_LAW6_dxsec_data["DDX"].NEin
        self.Ein_ddx = ENDF_LAW6_dxsec_data["DDX"].Ein
        self.Ncos_ddx = ENDF_LAW6_dxsec_data["DDX"].Ncos
        self.cos_ddx = ENDF_LAW6_dxsec_data["DDX"].cos
        self.NEout_ddx = ENDF_LAW6_dxsec_data["DDX"].NEout
        self.Eout_ddx = ENDF_LAW6_dxsec_data["DDX"].Eout
        self.f_ddx = ENDF_LAW6_dxsec_data["DDX"].f
        self.Emax_ddx = ENDF_LAW6_dxsec_data["DDX"].Emax

        # Build interpolator dict
        self.f_ddx_interp = {}
        for Ecounter in range(self.NEin_ddx):
            for Ccounter in range(self.Ncos_ddx[Ecounter]):
                self.f_ddx_interp[(Ecounter, Ccounter)] = interpolate_1d(
                    self.Eout_ddx[(Ecounter, Ccounter)],
                    self.f_ddx[(Ecounter, Ccounter)],
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )

    # Interpolate using Unit Base Transform
    def interpolate(self, Ein, mu, Eout):
        # Find indices
        # Energies
        if Ein == np.amax(self.Ein_ddx):
            Eidx2 = self.NEin_ddx - 1
            Eidx1 = Eidx2 - 1
        elif Ein < np.amin(self.Ein_ddx):
            return 0.0
        else:
            Eidx2 = np.argmax(self.Ein_ddx > Ein)
            Eidx1 = Eidx2 - 1
        # Angles
        if mu == +1.0:
            Cidx12 = self.Ncos_ddx[Eidx1] - 1
            Cidx11 = Cidx12 - 1
            Cidx22 = self.Ncos_ddx[Eidx2] - 1
            Cidx21 = Cidx22 - 1
        elif mu == -1.0:
            Cidx12 = 1
            Cidx11 = 0
            Cidx22 = 1
            Cidx21 = 0
        else:
            Cidx12 = np.argmax(self.cos_ddx[Eidx1] > mu)
            Cidx11 = Cidx12 - 1
            Cidx22 = np.argmax(self.cos_ddx[Eidx2] > mu)
            Cidx21 = Cidx22 - 1

        # Find interpolation factors
        mu_x1 = (mu - self.cos_ddx[Eidx1][Cidx11]) / (self.cos_ddx[Eidx1][Cidx12] - self.cos_ddx[Eidx1][Cidx11])
        mu_x2 = (mu - self.cos_ddx[Eidx2][Cidx21]) / (self.cos_ddx[Eidx2][Cidx22] - self.cos_ddx[Eidx2][Cidx21])
        Ein_x = (Ein - self.Ein_ddx[Eidx1]) / (self.Ein_ddx[Eidx2] - self.Ein_ddx[Eidx1])

        x_112 = mu_x1
        x_111 = 1 - x_112
        x_222 = mu_x2
        x_221 = 1 - x_222

        x_2 = Ein_x
        x_1 = 1 - x_2

        # Unit base transform
        E_h11 = self.Emax_ddx[(Eidx1, Cidx11)]
        E_h12 = self.Emax_ddx[(Eidx1, Cidx12)]
        E_h21 = self.Emax_ddx[(Eidx2, Cidx21)]
        E_h22 = self.Emax_ddx[(Eidx2, Cidx22)]
        E_h1 = E_h11 + mu_x1 * (E_h12 - E_h11)
        E_h2 = E_h21 + mu_x2 * (E_h22 - E_h21)
        E_high = E_h1 + Ein_x * (E_h2 - E_h1)
        if E_high == 0.0:
            return 0.0
        J_111 = self.Emax_ddx[(Eidx1, Cidx11)] / E_high
        J_112 = self.Emax_ddx[(Eidx1, Cidx12)] / E_high
        J_221 = self.Emax_ddx[(Eidx2, Cidx21)] / E_high
        J_222 = self.Emax_ddx[(Eidx2, Cidx22)] / E_high

        # Find unit base transformed energy
        Eout_111 = Eout * J_111
        Eout_112 = Eout * J_112
        Eout_221 = Eout * J_221
        Eout_222 = Eout * J_222

        f_111 = self.f_ddx_interp[(Eidx1, Cidx11)](Eout_111) * J_111
        f_112 = self.f_ddx_interp[(Eidx1, Cidx12)](Eout_112) * J_112
        f_221 = self.f_ddx_interp[(Eidx2, Cidx21)](Eout_221) * J_221
        f_222 = self.f_ddx_interp[(Eidx2, Cidx22)](Eout_222) * J_222

        f_1 = x_111 * f_111 + x_112 * f_112
        f_2 = x_221 * f_221 + x_222 * f_222

        f_ddx = x_1 * f_1 + x_2 * f_2

        return f_ddx

    def regular_grid(self, Ein, mu, Eout):
        self.rgrid_shape = (Ein.shape[0], mu.shape[0], Eout.shape[0])
        self.rgrid = np.zeros(self.rgrid_shape)
        for i in range(Ein.shape[0]):
            for j in range(mu.shape[0]):
                self.rgrid[i, j, :] = 2.0 * self.xsec_interp(Ein[i]) * self.interpolate(Ein[i], mu[j], Eout)


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

    def ddx(self, Ein, mu, Eout):
        E_star = Ein * self.A_i * self.A_e / (self.A_t + self.A_i) ** 2
        E_a = self.A_t * Ein / (self.A_p + self.A_t) + self.Q_react
        E_max = (self.A_tot - 1.0) * E_a / self.A_tot
        C3 = 4.0 / (np.pi * E_max * E_max)
        square_bracket_term = E_max - (E_star + Eout - 2 * mu * np.sqrt(E_star * Eout))
        square_bracket_term = np.where(square_bracket_term < 0.0, 0.0, square_bracket_term)
        f_ddx = C3 * np.sqrt(Eout * square_bracket_term)
        return f_ddx

    def regular_grid(self, Ein, mu, Eout):
        Ei, Mm, Eo = np.meshgrid(Ein, mu, Eout, indexing="ij")
        self.rgrid = 2.0 * self.xsec_interp(Ei) * self.ddx(Ei, Mm, Eo)
