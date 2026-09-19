# Backend of spectral model

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import NeSST.collisions as col
import NeSST.cross_sections as xs
from NeSST.constants import *
from NeSST.endf_interface import retrieve_ENDF_data
from NeSST.utils import *

##################
# Material class #
##################

# A values needed for scattering kinematics
A_H = Mp / Mn
A_D = Md / Mn
A_T = Mt / Mn
A_C = MC / Mn
A_Be = MBe / Mn


def unity(x):
    return jnp.ones_like(x)


class ElasticScatterKernel(eqx.Module):
    """Stationary ion elastic scattering matrices"""

    A: float
    dxs: xs.DifferentialCrossSection

    @eqx.filter_jit
    def __call__(self, Ein, Eout):
        Ei, Eo = jnp.meshgrid(Ein, Eout)
        muc = col.muc(self.A, Ei, Eo, 1.0, -1.0, 0.0)
        mu0 = col.mu_out(self.A, Ei, Eo, 0.0)
        dsdO = self.dxs(Ein, muc, Ei)
        jacob = col.g(self.A, Ei, Eo, 1.0, -1.0, 0.0)
        return mu0, jacob * dsdO


def affine_muc_band(muc_of_Eout, Eprobe):
    """Linearise the centre of mass cosine in Eout and invert it for mu_c = +-1

    mu_c is affine in Eout at fixed incoming energy, so this transform gives the
    Heaviside limits to integrate over the energy bins.

    Args:
        muc_of_Eout (callable): mu_c as a function of outgoing energy alone
        Eprobe (array): outgoing energies to linearise about

    Returns:
        tuple: the lower and upper band limits, and the value and slope of the
        linearisation about Eprobe
    """
    mu_probe, slope = jax.jvp(muc_of_Eout, (Eprobe,), (jnp.ones_like(Eprobe),))
    edge_a = Eprobe + (1.0 - mu_probe) / slope
    edge_b = Eprobe + (-1.0 - mu_probe) / slope
    return jnp.minimum(edge_a, edge_b), jnp.maximum(edge_a, edge_b), mu_probe, slope


class BinAveragedElasticScatterKernel(eqx.Module):
    """Elastic scattering kernel bin averaged over the energy grids"""

    A: float
    dxs: xs.DifferentialCrossSection
    # Midpoint sub-divisions per bin; the band edges are exact even at N = 1
    N: int = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, Ein, Eout):
        Ein_edges, _ = energy_bin_edges(Ein)
        Eout_edges, dEout = energy_bin_edges(Eout)

        # Incoming sub-nodes, flattened so the incoming axis is last as the
        # differential cross section evaluators expect
        Ei_sub, _ = midpoint_subnodes(Ein_edges[:-1], Ein_edges[1:], self.N)
        Ev = Ei_sub.reshape(-1)

        # Linearise about forward scatter, where mu_c is exactly +1
        band_lo, band_hi, mu_probe, slope = affine_muc_band(lambda Eo: col.muc(self.A, Ev, Eo, 1.0, -1.0, 0.0), Ev)

        # Overlap of the kinematically allowed band with each outgoing bin
        lo = jnp.maximum(Eout_edges[:-1, None], band_lo[None, :])
        hi = jnp.minimum(Eout_edges[1:, None], band_hi[None, :])
        Eo_sub, dEo_sub = midpoint_subnodes(lo, hi, self.N)
        Eo_sub = jnp.moveaxis(Eo_sub, -1, 1)
        dEo_sub = jnp.where(hi > lo, dEo_sub, 0.0)

        # Evaluating mu_c through the linearisation rather than col.muc also
        # avoids its removable singularity at Eout -> 0 for A = 1
        muc = mu_probe + slope * (Eo_sub - Ev)
        integrand = slope * self.dxs(Ev, muc, Ev)

        # Integrate over the outgoing bin, then average over the incoming bin
        inner = jnp.sum(integrand, axis=1) * dEo_sub
        inner = inner.reshape(inner.shape[0], -1, self.N).mean(axis=2)
        dNdEdmu = inner / dEout[:, None]

        # mu0 stays a point value per bin pair, clipped so that an edge bin with
        # a non-zero bin average cannot hand rhoL_func an out of range cosine
        Ei, Eo = jnp.meshgrid(Ein, Eout)
        mu0 = jnp.clip(col.mu_out(self.A, Ei, Eo, 0.0), -1.0, 1.0)
        return mu0, dNdEdmu


class InelasticScatterKernel(eqx.Module):
    """Stationary ion inelastic scattering matrices, classical kinematics"""

    A: float
    Q: float
    dxs: xs.DifferentialCrossSection

    @eqx.filter_jit
    def __call__(self, Ein, Eout):
        Ei, Eo = jnp.meshgrid(Ein, Eout)
        kin_a2 = (self.A / (self.A + 1)) ** 2 * (1.0 + (self.A + 1) / self.A * self.Q / Ei)
        kin_a2_safe = jnp.where(kin_a2 < 0.0, 1.0, kin_a2)
        kin_a = jnp.sqrt(kin_a2_safe)
        kin_b = 1.0 / (self.A + 1)
        muc = ((Eo / Ei) - kin_a**2 - kin_b**2) / (2 * kin_a * kin_b)
        mu0 = (jnp.sqrt(Eo / Ei) - (kin_a**2 - kin_b**2) * jnp.sqrt(Ei / Eo)) / (2 * kin_b)
        mu0 = jnp.where(kin_a2 < 0.0, 0.0, mu0)

        dsdO = self.dxs(Ein, muc, Ei)

        jacob = 2.0 / ((kin_a + kin_b) ** 2 - (kin_a - kin_b) ** 2) / Ei
        dNdEdmu = jnp.where(kin_a2 < 0.0, 0.0, jacob * dsdO)
        return mu0, dNdEdmu


class BinAveragedInelasticScatterKernel(eqx.Module):
    """Inelastic scattering kernel bin averaged over the energy grids"""

    A: float
    Q: float
    dxs: xs.DifferentialCrossSection
    N: int = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, Ein, Eout):
        Ein_edges, _ = energy_bin_edges(Ein)
        Eout_edges, dEout = energy_bin_edges(Eout)

        Ei_sub, _ = midpoint_subnodes(Ein_edges[:-1], Ein_edges[1:], self.N)
        Ev = Ei_sub.reshape(-1)

        kin_a2 = (self.A / (self.A + 1)) ** 2 * (1.0 + (self.A + 1) / self.A * self.Q / Ev)
        open_channel = kin_a2 >= 0.0
        kin_a = jnp.sqrt(jnp.where(open_channel, kin_a2, 1.0))
        kin_b = 1.0 / (self.A + 1)

        band_lo, band_hi, mu_probe, slope = affine_muc_band(
            lambda Eo: ((Eo / Ev) - kin_a**2 - kin_b**2) / (2 * kin_a * kin_b), Ev
        )
        # Below threshold the band is empty rather than kinematically bounded
        band_lo = jnp.where(open_channel, band_lo, 0.0)
        band_hi = jnp.where(open_channel, band_hi, 0.0)

        lo = jnp.maximum(Eout_edges[:-1, None], band_lo[None, :])
        hi = jnp.minimum(Eout_edges[1:, None], band_hi[None, :])
        Eo_sub, dEo_sub = midpoint_subnodes(lo, hi, self.N)
        Eo_sub = jnp.moveaxis(Eo_sub, -1, 1)
        dEo_sub = jnp.where(hi > lo, dEo_sub, 0.0)

        muc = mu_probe + slope * (Eo_sub - Ev)
        integrand = jnp.where(open_channel, slope * self.dxs(Ev, muc, Ev), 0.0)

        inner = jnp.sum(integrand, axis=1) * dEo_sub
        inner = inner.reshape(inner.shape[0], -1, self.N).mean(axis=2)
        dNdEdmu = inner / dEout[:, None]

        Ei, Eo = jnp.meshgrid(Ein, Eout)
        kin_a2_c = (self.A / (self.A + 1)) ** 2 * (1.0 + (self.A + 1) / self.A * self.Q / Ei)
        kin_a_c = jnp.sqrt(jnp.where(kin_a2_c < 0.0, 1.0, kin_a2_c))
        mu0 = (jnp.sqrt(Eo / Ei) - (kin_a_c**2 - kin_b**2) * jnp.sqrt(Ei / Eo)) / (2 * kin_b)
        mu0 = jnp.where(kin_a2_c < 0.0, 0.0, jnp.clip(mu0, -1.0, 1.0))
        return mu0, dNdEdmu


class IonKinematicScatterKernel(eqx.Module):
    """Elastic scattering matrices including the scattering ion velocity"""

    A: float
    dxs: xs.DifferentialCrossSection

    @eqx.filter_jit
    def __call__(self, Eout, vvec, Ein):
        Eo, vv, Ei = jnp.meshgrid(Eout, vvec, Ein, indexing="ij")
        # Reverse velocity direction so +ve vf is implosion
        # Choose this way round so vf is +ve if shell coming TOWARDS detector
        vf = -vv
        muout = col.mu_out(self.A, Ei, Eo, vf)
        jacob = col.g(self.A, Ei, Eo, 1.0, muout, vf)
        flux_change = col.flux_change(Ei, 1.0, vf)
        # Integrand of Eq. 8 in A. J. Crilly 2019 PoP
        dsigdOmega = xs.dsigdOmega(self.A, Ei, Eo, Ein, 1.0, muout, vf, self.dxs)
        return flux_change * dsigdOmega * jacob, muout


class BinAveragedIonKinematicScatterKernel(eqx.Module):
    """Ion velocity elastic scattering kernel bin averaged over the energy grids

    The jacobian carries the moving target flux correction, so it is no longer
    the slope of mu_c and is kept explicit.  N = 1 is recommended here to avoid
    a large memory cost.
    """

    A: float
    dxs: xs.DifferentialCrossSection
    N: int = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, Eout, vvec, Ein):
        Ein_edges, _ = energy_bin_edges(Ein)
        Eout_edges, dEout = energy_bin_edges(Eout)

        Ei_sub, _ = midpoint_subnodes(Ein_edges[:-1], Ein_edges[1:], self.N)
        Ev = Ei_sub.reshape(-1)
        n_out, n_v, n_in = Eout.shape[0], vvec.shape[0], Ev.shape[0]

        # Reverse velocity direction so +ve vf is implosion
        vf = -vvec[:, None]
        Eiv = jnp.broadcast_to(Ev[None, :], (n_v, n_in))

        band_lo, band_hi, _, _ = affine_muc_band(
            lambda Eo: col.muc(self.A, Eiv, Eo, 1.0, col.mu_out(self.A, Eiv, Eo, vf), vf), Eiv
        )

        lo = jnp.maximum(Eout_edges[:-1, None, None], band_lo[None, :, :])
        hi = jnp.minimum(Eout_edges[1:, None, None], band_hi[None, :, :])
        Eo_sub, dEo_sub = midpoint_subnodes(lo, hi, self.N)
        dEo_sub = jnp.where(hi > lo, dEo_sub, 0.0)
        # Fold the outgoing sub-node axis into the outgoing axis so that the
        # differential cross section evaluators still see a 3D cosine
        Eo_sub = jnp.moveaxis(Eo_sub, -1, 1).reshape(n_out * self.N, n_v, n_in)

        muout = col.mu_out(self.A, Ev, Eo_sub, vf)
        jacob = col.g(self.A, Ev, Eo_sub, 1.0, muout, vf)
        flux_change = col.flux_change(Ev, 1.0, vf)
        # Integrand of Eq. 8 in A. J. Crilly 2019 PoP
        dsigdOmega = xs.dsigdOmega(self.A, Ev, Eo_sub, Ev, 1.0, muout, vf, self.dxs)
        integrand = flux_change * dsigdOmega * jacob

        # Integrate over the outgoing bin, then average over the incoming bin
        inner = jnp.sum(integrand.reshape(n_out, self.N, n_v, n_in), axis=1) * dEo_sub
        inner = inner.reshape(n_out, n_v, -1, self.N).mean(axis=3)
        M = inner / dEout[:, None, None]

        Eo_c, vv_c, Ei_c = jnp.meshgrid(Eout, vvec, Ein, indexing="ij")
        mu = jnp.clip(col.mu_out(self.A, Ei_c, Eo_c, -vv_c), -1.0, 1.0)
        return M, mu


@eqx.filter_jit
def dNdE_integral(dNdEdmu, rhoL_asym, I_E, Ein):
    return jnp.trapezoid(dNdEdmu * rhoL_asym * I_E[None, :], Ein, axis=1)


@eqx.filter_jit
def dNdE_bin_integral(dNdEdmu, rhoL_asym, I_E, dEin):
    return jnp.sum(dNdEdmu * rhoL_asym * I_E[None, :] * dEin[None, :], axis=1)


@eqx.filter_jit
def n2n_dNdE_integral(rgrid, rhoL_asym, n2n_mu, I_E, Ein):
    grid_dNdE = jnp.trapezoid(rgrid * rhoL_asym[None, :, None], n2n_mu, axis=1)
    return jnp.trapezoid(I_E[:, None] * grid_dNdE, Ein, axis=0)


@eqx.filter_jit
def primspec_integral(rhoL_mult, full_scattering_M, I_E, Ein):
    return jnp.trapezoid(rhoL_mult * full_scattering_M * I_E[None, None, :], Ein, axis=2)


@eqx.filter_jit
def primspec_bin_integral(rhoL_mult, full_scattering_M, I_E, dEin):
    return jnp.sum(rhoL_mult * full_scattering_M * I_E[None, None, :] * dEin[None, None, :], axis=2)


@eqx.filter_jit
def gaussian_velocity_integral(M_prim, vvec, vbar, dv):
    gauss = jnp.exp(-((vvec - vbar) ** 2) / 2.0 / (dv**2)) / jnp.sqrt(2 * jnp.pi) / dv
    return jnp.trapezoid(M_prim * gauss[None, :], vvec, axis=1)


class material_data:
    def __init__(self, label, json):
        self.label = label

        self.json = json
        print(f">> NeSST: First usage of material {label}")
        print(f">> NeSST: Loading cross section data for {label}, with config {json}...")
        ENDF_data = retrieve_ENDF_data(self.json)

        self.A = ENDF_data["A"]

        if ENDF_data["interactions"].total:
            self.sigma_tot = interpolate_1d(
                ENDF_data["total_xsec"]["E"],
                ENDF_data["total_xsec"]["sig"],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

        if ENDF_data["interactions"].elastic:
            self.sigma = interpolate_1d(
                ENDF_data["elastic_xsec"]["E"],
                ENDF_data["elastic_xsec"]["sig"],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

            self.elastic_legendre = ENDF_data["elastic_dxsec"]["legendre"]
            if self.elastic_legendre:
                self.legendre_dx_spline = [unity]
                for i in range(ENDF_data["elastic_dxsec"]["N_l"]):
                    self.legendre_dx_spline.append(
                        interpolate_1d(
                            ENDF_data["elastic_dxsec"]["E"],
                            ENDF_data["elastic_dxsec"]["a_l"][:, i],
                            method="linear",
                            bounds_error=False,
                            fill_value=0.0,
                        )
                    )
                self.elastic_SDX_table = None
            else:
                self.legendre_dx_spline = None
                self.elastic_SDX_table = ENDF_data["elastic_dxsec"]["SDX"]

            self.elastic_dxs = xs.DifferentialCrossSection(
                sigma=self.sigma,
                legendre=tuple(self.legendre_dx_spline) if self.elastic_legendre else None,
                SDX=self.elastic_SDX_table,
            )

        self.l_n2n = ENDF_data["interactions"].n2n
        if ENDF_data["interactions"].n2n:
            if ENDF_data["n2n_dxsec"]["LAW"] == 6:
                self.n2n_ddx = xs.doubledifferentialcrosssection_LAW6(ENDF_data["n2n_xsec"], ENDF_data["n2n_dxsec"])
            elif ENDF_data["n2n_dxsec"]["LAW"] == 7:
                numerics = ENDF_data["numerics"]
                self.n2n_ddx = xs.doubledifferentialcrosssection_data(
                    ENDF_data["n2n_xsec"],
                    ENDF_data["n2n_dxsec"],
                    unit_base=numerics.law7_unit_base,
                    unit_base_N=numerics.law7_unit_base_N,
                )

        self.l_inelastic = ENDF_data["interactions"].inelastic
        if ENDF_data["interactions"].inelastic:
            self.n_inelastic = ENDF_data["n_inelastic"]

            self.isigma = []
            self.inelasticQ = []
            self.inelastic_legendre = []
            self.legendre_idx_spline = []
            self.inelastic_SDX_table = []
            self.inelastic_dxs = []

            for i_inelastic in range(self.n_inelastic):
                xsec_table = ENDF_data[f"inelastic_xsec_n{i_inelastic + 1}"]
                self.isigma.append(
                    interpolate_1d(
                        xsec_table["E"], xsec_table["sig"], method="linear", bounds_error=False, fill_value=0.0
                    )
                )

                dxsec_table = ENDF_data[f"inelastic_dxsec_n{i_inelastic + 1}"]
                self.inelasticQ.append(dxsec_table["Q"])

                self.inelastic_legendre.append(dxsec_table["legendre"])
                if dxsec_table["legendre"]:
                    idx_spline = [unity]
                    for i in range(dxsec_table["N_l"]):
                        idx_spline.append(
                            interpolate_1d(
                                dxsec_table["E"],
                                dxsec_table["a_l"][:, i],
                                method="linear",
                                bounds_error=False,
                                fill_value=0.0,
                            )
                        )
                    self.legendre_idx_spline.append(idx_spline)
                    self.inelastic_SDX_table.append(None)
                else:
                    self.legendre_idx_spline.append(None)
                    self.inelastic_SDX_table.append(dxsec_table["SDX"])

                self.inelastic_dxs.append(
                    xs.DifferentialCrossSection(
                        sigma=self.isigma[i_inelastic],
                        legendre=tuple(self.legendre_idx_spline[i_inelastic])
                        if self.inelastic_legendre[i_inelastic]
                        else None,
                        SDX=self.inelastic_SDX_table[i_inelastic],
                    )
                )

        self.Ein = None
        self.Eout = None
        self.vvec = None
        self.bin_average = False
        self.bin_average_N = 1

    ############################################
    # Stationary ion scattered spectral shapes #
    ############################################

    def init_energy_grids(self, Eout, Ein):
        self.Eout = Eout
        self.Ein = Ein

    def init_station_scatter_matrices(self, Nm=100, bin_average=False, bin_average_N=1):
        self.bin_average = bin_average
        self.bin_average_N = bin_average_N
        self.init_station_elastic_scatter()
        if self.l_n2n:
            self.init_n2n_ddxs(Nm)
        if self.l_inelastic:
            self.init_station_inelastic_scatter()

    # Elastic scatter matrix
    def init_station_elastic_scatter(self):
        if self.bin_average:
            kernel = BinAveragedElasticScatterKernel(A=self.A, dxs=self.elastic_dxs, N=self.bin_average_N)
        else:
            kernel = ElasticScatterKernel(A=self.A, dxs=self.elastic_dxs)
        self.elastic_mu0, self.elastic_dNdEdmu = kernel(jnp.asarray(self.Ein), jnp.asarray(self.Eout))

    # Inelastic scatter matrix
    # Currently uses classical kinematics
    def init_station_inelastic_scatter(self):
        self.inelastic_mu0 = []
        self.inelastic_dNdEdmu = []
        for i_inelastic in range(self.n_inelastic):
            Q, dxs = self.inelasticQ[i_inelastic], self.inelastic_dxs[i_inelastic]
            if self.bin_average:
                kernel = BinAveragedInelasticScatterKernel(A=self.A, Q=Q, dxs=dxs, N=self.bin_average_N)
            else:
                kernel = InelasticScatterKernel(A=self.A, Q=Q, dxs=dxs)
            mu0, dNdEdmu = kernel(jnp.asarray(self.Ein), jnp.asarray(self.Eout))
            self.inelastic_mu0.append(mu0)
            self.inelastic_dNdEdmu.append(dNdEdmu)

    def init_n2n_ddxs(self, Nm=100):
        self.n2n_mu = jnp.linspace(-1.0, 1.0, Nm)
        self.n2n_ddx.regular_grid(self.Ein, self.n2n_mu, self.Eout)

    def calc_dNdEs(self, I_E, rhoL_func):
        self.calc_station_elastic_dNdE(I_E, rhoL_func)
        if self.l_n2n:
            self.calc_n2n_dNdE(I_E, rhoL_func)
        if self.l_inelastic:
            self.calc_station_inelastic_dNdE(I_E, rhoL_func)

    # Spectrum produced by scattering of incoming isotropic neutron source I_E with normalised areal density asymmetry rhoR_asym_func
    def calc_station_elastic_dNdE(self, I_E, rhoL_func):
        rhoL_asym = rhoL_func(self.elastic_mu0)
        # A bin averaged matrix is contracted as a bin sum, not trapezoided
        if self.bin_average:
            _, dEin = energy_bin_edges(jnp.asarray(self.Ein))
            self.elastic_dNdE = dNdE_bin_integral(self.elastic_dNdEdmu, rhoL_asym, I_E, dEin)
        else:
            self.elastic_dNdE = dNdE_integral(self.elastic_dNdEdmu, rhoL_asym, I_E, jnp.asarray(self.Ein))

    def calc_station_inelastic_dNdE(self, I_E, rhoL_func):
        self.inelastic_dNdE = jnp.zeros(self.Eout.shape[0])
        _, dEin = energy_bin_edges(jnp.asarray(self.Ein))
        for i_inelastic in range(self.n_inelastic):
            rhoL_asym = rhoL_func(self.inelastic_mu0[i_inelastic])
            dNdEdmu = self.inelastic_dNdEdmu[i_inelastic]
            if self.bin_average:
                self.inelastic_dNdE += dNdE_bin_integral(dNdEdmu, rhoL_asym, I_E, dEin)
            else:
                self.inelastic_dNdE += dNdE_integral(dNdEdmu, rhoL_asym, I_E, jnp.asarray(self.Ein))

    def calc_n2n_dNdE(self, I_E, rhoL_func):
        rhoL_asym = rhoL_func(self.n2n_mu)
        self.n2n_dNdE = n2n_dNdE_integral(self.n2n_ddx.rgrid, rhoL_asym, self.n2n_mu, I_E, jnp.asarray(self.Ein))

    def rhoR_2_A1s(self, rhoR):
        mbar = self.A * Mn_kg
        A_1S = rhoR * (sigmabarn / mbar)
        return A_1S

    # # Spectrum produced by scattering of incoming neutron source with anisotropic birth spectrum
    # def elastic_scatter_aniso(self,Eout,Ein,mean_iso,mean_aniso,var_iso,b_spec,rhoR_asym_func):
    #     Ei,Eo  = np.meshgrid(Ein,Eout)
    #     muc    = col.muc(self.A,Ei,Eo,1.0,-1.0,0.0)
    #     sigma  = sigma_nT(Ein)
    #     E_vec  = Ein
    #     Tlcoeff,Nl     = interp_Tlcoeff(self.legendre_dx_spline,E_vec)
    #     Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    #     mu0 = col.mu_out(self.A,Ei,Eo,0.0)
    #     rhoR_asym = rhoR_asym_func(mu0)
    #     prim_mean = mean_iso+mean_aniso*mu0
    #     I_E_aniso = b_spec(Ei,prim_mean,var_iso)
    #     dsdO = diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
    #     jacob = col.g(self.A,Ei,Eo,1.0,-1.0,0.0)
    #     res = np.trapezoid(jacob*dsdO*I_E_aniso*rhoR_asym,Ein,axis=-1)
    #     return res

    #####################################################
    # Inclusion of ion velocities to scattering kernels #
    #####################################################
    def full_scattering_matrix_create(self, vvec, bin_average_N=1):
        self.vvec = vvec

        if self.bin_average:
            kernel = BinAveragedIonKinematicScatterKernel(A=self.A, dxs=self.elastic_dxs, N=bin_average_N)
        else:
            kernel = IonKinematicScatterKernel(A=self.A, dxs=self.elastic_dxs)
        self.full_scattering_M, self.full_scattering_mu = kernel(
            jnp.asarray(self.Eout), jnp.asarray(vvec), jnp.asarray(self.Ein)
        )
        self.rhoL_mult = jnp.ones_like(self.full_scattering_mu)

    def scattering_matrix_apply_rhoLfunc(self, rhoL_func):
        # Find multiplicative factor for areal density asymmetries
        self.rhoL_mult = rhoL_func(self.full_scattering_mu)

    # Integrate out the birth neutron spectrum
    def matrix_primspec_int(self, I_E):
        if self.bin_average:
            _, dEin = energy_bin_edges(jnp.asarray(self.Ein))
            self.M_prim = primspec_bin_integral(self.rhoL_mult, self.full_scattering_M, I_E, dEin)
        else:
            self.M_prim = primspec_integral(self.rhoL_mult, self.full_scattering_M, I_E, jnp.asarray(self.Ein))

    # Integrate out the ion velocity distribution
    def matrix_interpolate_gaussian(self, E, vbar, dv):
        # Integrating over Gaussian
        M_v = gaussian_velocity_integral(self.M_prim, jnp.asarray(self.vvec), jnp.asarray(vbar), jnp.asarray(dv))
        # Interpolate to energy points E
        interp = interpolate_1d(self.Eout, M_v, method="linear", bounds_error=False)
        return interp(E)


class TT_spectrum_model(eqx.Module):
    TT_spec_E: Array
    CoM_E_Brune: Array
    CoM_spec_Brune: Array
    CoM_E_Eriksson: Array
    CoM_spec_Eriksson: Array
    CoM_E_GJ_low: Array
    CoM_spec_GJ_low: Array
    CoM_E_GJ_mid: Array
    CoM_spec_GJ_mid: Array
    CoM_E_GJ_high: Array
    CoM_spec_GJ_high: Array
    TT_reac_McNally_spline: Interpolator1D
    TT_reac_Hale_spline: Interpolator1D
    available_spectrum_models: list = eqx.field(static=True)
    available_reactivity_models: list = eqx.field(static=True)

    def __init__(self, NE=500):
        # Create TT spectrum model grid
        self.TT_spec_E = jnp.linspace(1e-10, 12e6, NE)  # eV

        # Load TT spectra (CoM frame)
        self.CoM_E_Brune, self.CoM_spec_Brune = self._load_and_normalise_CoM_spec(data_dir + "TT/BruneFit16_36keV.txt")
        self.CoM_E_Eriksson, self.CoM_spec_Eriksson = self._load_and_normalise_CoM_spec(
            data_dir + "TT/Eriksson_45keV.txt"
        )
        self.CoM_E_GJ_low, self.CoM_spec_GJ_low = self._load_and_normalise_CoM_spec(
            data_dir + "TT/GatuJohnson_16keV.txt"
        )
        self.CoM_E_GJ_mid, self.CoM_spec_GJ_mid = self._load_and_normalise_CoM_spec(
            data_dir + "TT/GatuJohnson_36keV.txt"
        )
        self.CoM_E_GJ_high, self.CoM_spec_GJ_high = self._load_and_normalise_CoM_spec(
            data_dir + "TT/GatuJohnson_50keV.txt"
        )

        self.available_spectrum_models = [
            "Brune",
            "Eriksson",
            "Gatu-Johnson-low",
            "Gatu-Johnson-mid",
            "Gatu-Johnson-high",
        ]

        # Load TT reactivity
        TT_reac_McNally_data = np.loadtxt(
            data_dir + "TT/TT_reac_McNally.dat"
        )  # sigmav im m^3/s   # From https://www.osti.gov/servlets/purl/5992170 - N.B. not in agreement with experimental measurements
        self.TT_reac_McNally_spline = interpolate_1d(
            TT_reac_McNally_data[:, 0], TT_reac_McNally_data[:, 1], method="linear", bounds_error=False, fill_value=0.0
        )
        TT_reac_Hale_data = np.loadtxt(data_dir + "TT/TT_reac_Hale.dat")  # T in MeV, sigmav im cm^3/s   # From Hale
        self.TT_reac_Hale_spline = interpolate_1d(
            TT_reac_Hale_data[:, 0] * 1e3,
            TT_reac_Hale_data[:, 1] * 1e-6,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        # TT_reac_data = np.loadtxt(data_dir + "TT_reac_ENDF.dat")       # sigmav im m^3/s   # From ENDF
        # TT_reac_spline = interpolate_1d(TT_reac_data[:,0],TT_reac_data[:,1],method='linear',bounds_error=False,fill_value=0.0)

        self.available_reactivity_models = ["Hale", "McNally", "CaughlanFowler"]

    def _load_and_normalise_CoM_spec(self, filename):
        E, spec = np.loadtxt(filename, unpack=True)
        E = E * 1e6  # MeV to eV
        # Shift 0
        E[0] += 1e-10
        # Interpolate to model energy grid
        spec = jnp.interp(self.TT_spec_E, E, spec, left=0.0, right=0.0)
        spec = spec / jnp.trapezoid(spec, self.TT_spec_E)  # Normalise to 1
        return self.TT_spec_E, spec

    def reac(self, Ti, model):
        Ti_kev = Ti / 1e3
        if model == "Hale":
            return self.TT_reac_Hale_spline(Ti_kev)
        elif model == "McNally":
            return self.TT_reac_McNally_spline(Ti_kev)
        elif model == "CaughlanFowler":
            T9 = (Ti_kev * sc.e * 1e3 / sc.k) / 1e9
            T9_1third = T9 ** (1.0 / 3.0)
            poly = jnp.polyval(jnp.array([0.225, 0.148, -0.272, -0.455, 0.086, 1.0]), T9_1third)
            return (1 / sc.N_A) * 1.67e3 / T9 ** (2.0 / 3.0) * jnp.exp(-4.872 / T9 ** (1.0 / 3.0)) * poly
        else:
            print(f"WARNING: TT model name ({model}) not recognised! Default to 0")
            return jnp.zeros_like(Ti)

    def spec(self, E, Ti, model):
        if model == "Brune":
            CoM_E, CoM_spec = self.CoM_E_Brune, self.CoM_spec_Brune
        elif model == "Gatu-Johnson-low":
            CoM_E, CoM_spec = self.CoM_E_GJ_low, self.CoM_spec_GJ_low
        elif model == "Gatu-Johnson-mid":
            CoM_E, CoM_spec = self.CoM_E_GJ_mid, self.CoM_spec_GJ_mid
        elif model == "Gatu-Johnson-high":
            CoM_E, CoM_spec = self.CoM_E_GJ_high, self.CoM_spec_GJ_high
        elif model == "Eriksson":
            CoM_E, CoM_spec = self.CoM_E_Eriksson, self.CoM_spec_Eriksson
        else:
            print(f"WARNING: TT spectrum model name ({model}) not recognised! Default to 0")
            return jnp.zeros_like(E)

        sqrt_Ep1 = jnp.sqrt(E)
        sqrt_Ep2 = jnp.sqrt(CoM_E)
        dE = CoM_E[1] - CoM_E[0]

        # Following Appelbe HEDP 2016
        # https://www.sciencedirect.com/science/article/pii/S1574181816300295
        int_factor = jnp.exp(-2 * Mt / Mn / Ti * (sqrt_Ep1[:, None] - sqrt_Ep2[None, :]) ** 2) / sqrt_Ep2[None, :] * dE
        norm_factor = 0.5 * jnp.sqrt((2 * Mt / Mn / Ti) / jnp.pi)

        integrand = norm_factor * int_factor * CoM_spec[None, :]

        broadened_spec = jnp.sum(integrand, axis=1)

        return broadened_spec


TT_model = TT_spectrum_model()

########################
# Primary reactivities #
########################

# References:
# Bosch Hale: https://doi.org/10.1088/0029-5515/33/12/513
# Caughlan & Fowler: https://doi.org/10.1146/annurev.aa.13.090175.000441
# McNally: https://www.osti.gov/servlets/purl/5992170


# Output in m3/s, Ti in eV
def reac_DT(Ti, model="BoschHale"):
    Ti_kev = Ti / 1e3
    if model == "BoschHale":
        # Bosch Hale DT and DD reactivities
        # Taken from Atzeni & Meyer ter Vehn page 19
        C1 = 643.41e-22
        xi = 6.6610 * Ti_kev ** (-0.333333333)
        eta = 1 - jnp.polyval(jnp.array([-0.10675e-3, 4.6064e-3, 15.136e-3, 0.0e0]), Ti_kev) / jnp.polyval(
            jnp.array([0.01366e-3, 13.5e-3, 75.189e-3, 1.0e0]), Ti_kev
        )
        return C1 * eta ** (-0.833333333) * xi**2 * jnp.exp(-3 * eta ** (0.333333333) * xi)
    elif model == "CaughlanFowler":
        T9 = (Ti_kev * sc.e * 1e3 / sc.k) / 1e9
        T9_1third = T9 ** (1.0 / 3.0)
        poly = jnp.polyval(jnp.array([17.24, 10.52, 1.16, 1.80, 0.092, 1.0]), T9_1third)
        return (
            (1 / sc.N_A)
            * (8.09e4 * poly * jnp.exp(-4.524 / T9 ** (1.0 / 3.0) - (T9 / 0.120) ** 2) + 8.73e2 * jnp.exp(-0.523 / T9))
            / T9 ** (2.0 / 3.0)
        )
    else:
        print(f"WARNING: DT model name ({model}) not recognised! Default to 0")
        return jnp.zeros_like(Ti)


def reac_DD(Ti, model="BoschHale"):
    Ti_kev = Ti / 1e3
    if model == "BoschHale":
        # Bosch Hale DT and DD reactivities
        # Taken from Atzeni & Meyer ter Vehn page 19
        C1 = 3.5741e-22
        xi = 6.2696 * Ti_kev ** (-0.333333333)
        eta = 1 - jnp.polyval(jnp.array([5.8577e-3, 0.0e0]), Ti_kev) / jnp.polyval(
            jnp.array([-0.002964e-3, 7.6822e-3, 1.0e0]), Ti_kev
        )
        return C1 * eta ** (-0.833333333) * xi**2 * jnp.exp(-3 * eta ** (0.333333333) * xi)
    elif model == "CaughlanFowler":
        T9 = (Ti_kev * sc.e * 1e3 / sc.k) / 1e9
        T9_1third = T9 ** (1.0 / 3.0)
        poly = jnp.polyval(jnp.array([-0.071, -0.041, 0.6, 0.876, 0.098, 1.0]), T9_1third)
        return (1 / sc.N_A) * 3.97e2 / T9 ** (2.0 / 3.0) * jnp.exp(-4.258 / T9 ** (1.0 / 3.0)) * poly
    else:
        print(f"WARNING: DD model name ({model}) not recognised! Default to 0")
        return jnp.zeros_like(Ti)


def reac_TT(Ti, model="Hale"):
    return TT_model.reac(Ti, model=model)
