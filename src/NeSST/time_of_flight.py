from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.ndimage import uniform_filter1d
from scipy.special import erf

from NeSST.collisions import *
from NeSST.constants import *
from NeSST.endf_interface import retrieve_total_cross_section_from_ENDF_file
from NeSST.utils import *


@dataclass
class LOS_material_component:
    number_fraction: float
    A: float
    ENDF_file: str


@dataclass
class LOS_material:
    density: float
    ntot: float = field(init=False)
    mavg: float = field(init=False)
    length: float
    components: List[LOS_material_component]

    def __post_init__(self):
        self.mavg = 0.0
        for comp in self.components:
            self.mavg += comp.number_fraction * comp.A
        self.mavg *= sc.atomic_mass
        self.ntot = self.density / self.mavg


def get_LOS_attenuation(LOS_materials: List[LOS_material]):
    tau_interp_list = []
    for LOS_material in LOS_materials:
        ntot_barn = 1e-28 * LOS_material.ntot
        L = LOS_material.length
        for LOS_component in LOS_material.components:
            ncomp = ntot_barn * LOS_component.number_fraction
            E, sigma_tot = retrieve_total_cross_section_from_ENDF_file(LOS_component.ENDF_file)
            tau_interp = interpolate_1d(E, L * ncomp * sigma_tot, method="linear")
            tau_interp_list.append(tau_interp)

    def LOS_attenuation(E):
        total_tau = tau_interp_list[0](E)
        for i in range(1, len(tau_interp_list)):
            total_tau += tau_interp_list[i](E)
        transmission = np.exp(-total_tau)
        return transmission

    return LOS_attenuation


class ProtonScintillationModel(ABC):
    """
    Using equation (3) from

    Qi Tang, Zifeng Song, Pinyang Liu, Bo Yu, Jiamin Yang,
    Calibration of the sensitivity of the bibenzyl-based scintillation detector to 1–5 MeV neutrons,
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment,
    Volume 1068,
    2024,
    169779,
    ISSN 0168-9002,
    https://doi.org/10.1016/j.nima.2024.169779.
    """

    def __init__(self, Enorm):
        self.Enorm = Enorm
        self.normalisation = self._unnormalised(self.Enorm)

    @abstractmethod
    def L_integral(self, E):
        pass

    def _unnormalised(self, E):
        return mat_dict["H"].sigma(E) * self.L_integral(E) / E

    def __call__(self, E):
        return self._unnormalised(E) / self.normalisation


class PowerLawScintillationModel(ProtonScintillationModel):
    def __init__(self, p, Enorm):
        self.p = p
        super().__init__(Enorm)

    def L_integral(self, E):
        return E ** (self.p + 1) / (self.p + 1)


class VerbinskiNLOModel(ProtonScintillationModel):
    """
    Data from Verbinski, VVl, et al.
    "Calibration of an organic scintillator for neutron spectrometry."
    Nuclear Instruments and Methods 65.1 (1968): 8-25.
    """

    def __init__(self, Enorm):
        V_E, V_L = np.loadtxt(data_dir + "VerbinskiLproton.csv", delimiter=",", unpack=True)
        cumulative_L = cumtrapz(y=np.insert(V_L, 0, 0.0), x=np.insert(V_E, 0, 0.0))
        self.L_integral_interp = interpolate_1d(
            np.insert(V_E, 0, 0.0) * 1e6, np.insert(cumulative_L, 0, 0.0), method="cubic"
        )
        super().__init__(Enorm)

    def L_integral(self, E):
        return self.L_integral_interp(E)


class BirksBetheBlochNLOModel(ProtonScintillationModel):
    r"""
    akB in eV

    Combining:
    kB in m/eV
    a in eV^2/m
    Bethe-Bloch formula for stopping power:
        dE/dx = a/E

    Birks relation for light response:
        dL/dx \propto (dEdx)/(1+kB (dE/dx))
    """

    def __init__(self, akB, Enorm):
        self.akB = akB
        super().__init__(Enorm)

    def L_integral(self, E):
        return 0.5 * E**2 - self.akB * E - self.akB * (self.akB + E) * np.log(1.0 + E / self.akB)


class BirksBetheNLOModel(ProtonScintillationModel):
    r"""
    akB in eV

    Combining:
    kB in m/eV
    a in eV^2/m

    I = excitation_energy in eV

    Bethe formula for stopping power:
        dE/dx = a/E * ln(4 me E / mp I)

    Birks relation for light response:
        dL/dx \propto (dEdx)/(1+kB (dE/dx))
    """

    def __init__(self, akB, excitation_energy, mp, Enorm, Emin, Emax, NE_interp):
        self.akB = akB
        self.excitation_energy = excitation_energy
        self.mp = mp

        self.Istar = excitation_energy * mp / sc.m_e / 4.0

        # Precompute tables for the integral and L(E) to speed up the interpolation
        E_grid = np.linspace(Emin, Emax, NE_interp)
        dLdE_grid = self.dLdE(E_grid)
        L_grid = cumtrapz(y=dLdE_grid, x=E_grid, initial=0.0)
        self.L_interp = interpolate_1d(E_grid, L_grid, method="cubic")
        L_integral_grid = cumtrapz(y=L_grid, x=E_grid, initial=0.0)
        self.L_integral_interp = interpolate_1d(E_grid, L_integral_grid, method="cubic")
        super().__init__(Enorm)

    def kB_dEdx(self, Ep):
        Ep_lim = np.maximum(Ep, np.e * self.Istar)
        return self.akB / Ep_lim * np.log(Ep_lim / self.Istar)

    def dLdE(self, Ep):
        return 1.0 / (1.0 + self.kB_dEdx(Ep))

    def L(self, E):
        return self.L_interp(E)

    def L_integral(self, E):
        return self.L_integral_interp(E)


class CraunSmithBetheModel(BirksBetheNLOModel):
    r"""
    Craun, R. L., and D. L. Smith.
    "Analysis of response data for several organic scintillators."
    Nuclear Instruments and Methods 80.2 (1970): 239-244.

    We use:

    dLdE \propto 1 / (1 + kB * dE/dx + C (kb * dE/dx)**2)

    where dE/dx is given by the Bethe formula.

    Note that this is a slight redefinition of C compared with the original paper,
    which is more convenient for our purposes (dimensionless).

    """

    def __init__(self, C, akB, excitation_energy, mp, Enorm, Emin, Emax, NE_interp):
        self.C = C
        super().__init__(akB, excitation_energy, mp, Enorm, Emin, Emax, NE_interp)

    def dLdE(self, Ep):
        return 1.0 / (1.0 + self.kB_dEdx(Ep) + self.C * (self.kB_dEdx(Ep) ** 2))


# Getters for common scintillation models, for ease of use in nToF class
get_power_law_NLO = lambda p, Enorm=E0_DT: PowerLawScintillationModel(p, Enorm)

get_Verbinski_NLO = lambda Enorm=E0_DT: VerbinskiNLOModel(Enorm)

get_BirksBetheBloch_NLO = lambda akB, Enorm=E0_DT: BirksBetheBlochNLOModel(akB, Enorm)

get_BirksBethe_NLO = (
    lambda akB, excitation_energy, mp=sc.m_p, Enorm=E0_DT, Emin=1e3, Emax=20e6, NE_interp=1000: BirksBetheNLOModel(
        akB, excitation_energy, mp, Enorm, Emin, Emax, NE_interp
    )
)

get_CraunSmithBethe_NLO = (
    lambda C, akB, excitation_energy, mp=sc.m_p, Enorm=E0_DT, Emin=1e3, Emax=20e6, NE_interp=1000: CraunSmithBetheModel(
        C, akB, excitation_energy, mp, Enorm, Emin, Emax, NE_interp
    )
)


def get_unity_sensitivity():
    def unity_sensitivity(En):
        return np.ones_like(En)

    return unity_sensitivity


def combine_detector_sensitivities(model_list):
    def total_sensitivity(E):
        sensitivity = model_list[0](E)
        for i in range(1, len(model_list)):
            sensitivity *= model_list[i](E)
        return sensitivity

    return total_sensitivity


def top_hat(scint_thickness):
    """
    Return a function  R_base(t_detected, En)  that creates the
    normalised top-hat transit matrix for *this* scintillator thickness.
    """

    def _top_hat_matrix(t_detected, t_transit):
        """NxN top-hat response, normalised row-wise."""
        tt_d, tt_a = np.meshgrid(t_detected, t_detected, indexing="ij")
        _, tt_t = np.meshgrid(t_detected, t_transit, indexing="ij")

        R = np.eye(t_detected.size) + np.heaviside(tt_d - tt_a, 0.0) - np.heaviside(tt_d - (tt_a + tt_t), 1.0)

        row_sum = R.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1  # avoid div-by-zero
        return R / row_sum

    def base(t_detected, En):
        vn = Ekin_2_beta(En, Mn) * c
        t_transit = scint_thickness / vn
        return _top_hat_matrix(t_detected, t_transit)

    return base


def inversegaussian_nIRF(
    scint_thickness, ni_scin=8.79e28, CH_ratio=8 / 18, E_lower=0.05e6, E_upper=25.0e6, NE_interp=1000
):
    if scint_thickness != 10e-2 or ni_scin != 8.79e28:
        raise NotImplementedError("Current inverse gaussian nIRF fit coefficients for 8.79e28 1/m^3 and 10cm only!")

    E_range = np.linspace(E_lower, E_upper, NE_interp)
    sig_H = mat_dict["H"].sigma_tot(E_range)
    sig_C = mat_dict["C12"].sigma_tot(E_range)

    def sig_CH(E):
        sig_barns = CH_ratio * np.interp(E, E_range, sig_C) + (1 - CH_ratio) * np.interp(E, E_range, sig_H)
        return sig_barns * 1e-28

    def IGtail_nIRF_bestfit_coeffs(En):
        """
        Best fit coefficients as a function of neutron energy

        NB these are specific to a scintillator geometry
        For different geometry you must re-perform MCNP/Geant/Scintillator1DMonteCarlo sims

        Analysis performed by A. Crilly, 2025
        """
        E_MeV = En / 1e6
        f = 0.46 * np.ones_like(E_MeV)
        A = 0.25 * np.ones_like(E_MeV)

        mu_inverse_ns = [
            0.45918122858399824,
            0.6557307634639333,
            0.8333427566991481,
            0.8986859864960112,
            1.146202470556479,
            1.2341761264319626,
            1.2686641782526762,
            2.8986506749332044,
            1.4799758893943886,
            2.3823877852359687,
            3.84910816578929,
            3.1966724468523355,
            4.284122083989886,
            3.452849096581845,
            4.838842934652534,
            9.999999999999998,
        ]

        lamb_inverse_ns = [
            0.7552223497006166,
            1.0369636485550817,
            1.1703584280444446,
            1.4085037422394058,
            1.445604993812616,
            1.464511388605816,
            1.6795667261078404,
            1.3844968569428273,
            1.7564016389637123,
            1.6801129259460668,
            1.6345197013154014,
            1.7884346810613823,
            1.7955059618288889,
            1.8762280752322453,
            1.9247350785402442,
            1.8572104736139436,
        ]
        Egrid = 1.0 + np.arange(len(mu_inverse_ns))

        mu = np.interp(E_MeV, Egrid, mu_inverse_ns) * 1e9
        lamb = np.interp(E_MeV, Egrid, lamb_inverse_ns) * 1e9
        return f, A, mu, lamb

    def base(t_detected, En):
        vn = Ekin_2_beta(En, Mn) * c
        t_transit = scint_thickness / vn

        tt_d, tt_a = np.meshgrid(t_detected, t_detected, indexing="ij")
        _, tt_t = np.meshgrid(t_detected, t_transit, indexing="ij")

        f, A, mu, lamb = IGtail_nIRF_bestfit_coeffs(En)

        top_hat_mat = np.eye(t_detected.size) + np.heaviside(tt_d - tt_a, 0.0) - np.heaviside(tt_d - (tt_a + tt_t), 1.0)
        exp_E_arg = f * vn * ni_scin * sig_CH(En)
        main_response = np.exp(-(tt_d - tt_a) * exp_E_arg[None, :]) * top_hat_mat

        t_shift = tt_d - (tt_a + tt_t)
        tail_hat = np.heaviside(t_shift, 0.5)
        t_shift[t_shift < 0.0] = 0.0
        prefactor = lamb / mu
        t_coeff = 2 * mu**2 / lamb
        exp_arg = prefactor[None, :] * (1 - np.sqrt(1 + t_coeff[None, :] * t_shift))
        tail_response = tail_hat * A[None, :] * np.exp(exp_arg)

        R = main_response + tail_response

        row_sum = R.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1  # avoid div-by-zero
        return R / row_sum

    return base


def decaying_gaussian_kernel(FWHM, tau, shift_sigma=2.0):
    """Single exponential tail multiplied by Gaussian."""
    sig = FWHM / 2.355
    shift_t = shift_sigma * sig

    def kernel(t):
        t_shift = t - shift_t
        erf_arg = (t_shift - sig**2 / tau) / np.sqrt(2 * sig**2)
        g = np.exp(-t_shift / tau) * np.exp(0.5 * sig**2 / tau**2)
        g *= (1 + erf(erf_arg)) / (2 * tau)
        g[t < 0] = 0
        return g / np.trapezoid(g, x=t)

    return kernel


def double_decay_gaussian_kernel(FWHM, taus, frac, shift_sigma=2.0):
    """Weighted sum of two decaying-Gaussian components."""
    k1 = decaying_gaussian_kernel(FWHM, taus[0], shift_sigma)
    k2 = decaying_gaussian_kernel(FWHM, taus[1], shift_sigma)

    def kernel(t):
        out = frac * k1(t) + (1 - frac) * k2(t)
        return out / np.trapezoid(out, x=t)

    return kernel


def gated_decaying_gaussian_kernel(sig, tau, shift_t, sig_turnon):
    """Exponentially decaying Gaussian with logistic gate."""

    def gate(x):
        g = np.where(x > 0, 2 / (1 + np.exp(-x)) - 1, 0)
        return g

    def kernel(t):
        t_shift = t - shift_t
        erf_arg = (t_shift - sig**2 / tau) / np.sqrt(2 * sig**2)
        g = np.exp(-t_shift / tau) * np.exp(0.5 * sig**2 / tau**2)
        g *= (1 + erf(erf_arg)) / (2 * tau)
        g *= gate(t / sig_turnon)
        return g / np.trapezoid(g, x=t)

    return kernel


def t_gaussian_kernel(FWHM, peak_pos):
    """t·Gaussian (often used for leading-edge shaping)."""
    sig = FWHM / 2.355
    mu = (peak_pos**2 - sig**2) / peak_pos

    def kernel(t):
        g = t * np.exp(-0.5 * ((t - mu) / sig) ** 2)
        g[t < 0] = 0
        return g / np.trapezoid(g, x=t)

    return kernel


def delta_kernel():
    return lambda t: np.array([1.0])


def make_transit_time_IRF(thickness, kernel_fn, base_matrix_fn=None):
    """
    Parameters
    ----------
    thickness : float
        Sets the detector thickness
    kernel_fn : callable(t) -> 1-D array
        Builds the convolution kernel on the *same* time grid.
    base_matrix_fn : callable(thickness) -> callable(t_detected, En) -> 2-D array  [optional]
        Anything that returns an (N×N) response matrix *before* filtering.
        If omitted, we fall back to the canonical top-hat.
    """
    # Fallback to the usual top-hat if the caller doesn't supply one
    if base_matrix_fn is None:
        base_matrix_fn = top_hat(thickness)
    else:
        base_matrix_fn = base_matrix_fn(thickness)

    def irf(t_detected, En):
        Rbase = base_matrix_fn(t_detected, En)
        kernel = kernel_fn(t_detected - 0.5 * (t_detected[-1] + t_detected[0]))

        Rconv = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=Rbase)

        row_sum = Rconv.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        return Rconv / row_sum

    return irf


def _roll_zero(arr, n):
    """Roll a 1-D array by ``n`` positions, filling vacated entries with zero.

    Unlike ``np.roll``, this does not wrap around.  Positive ``n`` shifts
    towards later times; negative ``n`` shifts towards earlier times.
    Shifts larger than the array length return an all-zero array.
    """
    out = np.zeros_like(arr)
    if n == 0:
        out[:] = arr
    elif n > 0:
        if n < len(arr):
            out[n:] = arr[:-n]
    else:  # n < 0
        if -n < len(arr):
            out[:n] = arr[-n:]
    return out


class nToF:
    def __init__(
        self,
        distance,
        sensitivity,
        instrument_response_function,
        normtime_start=5.0,
        normtime_end=20.0,
        normtime_N=2048,
        detector_normtime=None,
    ):
        self.distance = distance
        self.sensitivity = sensitivity
        self.instrument_response_function = instrument_response_function

        if detector_normtime is None:
            self.detector_normtime = np.linspace(normtime_start, normtime_end, normtime_N)
        else:
            self.detector_normtime = detector_normtime
        self.detector_time = self.detector_normtime * self.distance / c
        # Init instrument response values
        self.compute_instrument_response()

    def compute_instrument_response(self):
        self.En_det = beta_2_Ekin(1.0 / self.detector_normtime, Mn)

        self.dEdt = Jacobian_dEdnorm_t(self.En_det, Mn)
        self.sens = self.sensitivity(self.En_det)
        self.R = self.instrument_response_function(self.detector_time, self.En_det)

    def get_dNdt(self, En, dNdE):
        dNdE_interp = np.interp(self.En_det, En, dNdE, left=0.0, right=0.0)
        return dNdE_interp * self.dEdt

    def get_signal(self, En, dNdE):
        dNdt = self.get_dNdt(En, dNdE)

        return self.detector_time, self.detector_normtime, np.matmul(self.R, self.sens * dNdt)

    def get_signal_no_IRF(self, En, dNdE):
        dNdt = self.get_dNdt(En, dNdE)

        return self.detector_time, self.detector_normtime, dNdt

    # ------------------------------------------------------------------
    # Time-resolved (emission-time-dependent) methods
    # ------------------------------------------------------------------

    def get_time_resolved_dNdt(self, En, d2NdEdt):
        """Interpolate d²N/dE dt_emit onto the detector energy grid and apply
        the energy-to-normtime Jacobian.

        Parameters
        ----------
        En : array_like, shape (N_E,)
            Energy bin centres (eV).  Must be sorted ascending.
        d2NdEdt : array_like, shape (N_E, N_temit)
            Double-differential spectrum d²N/dE dt_emit  [1/eV/s].

        Returns
        -------
        dNdt2d : ndarray, shape (N_td, N_temit)
            d²N/dt_norm dt_emit on the detector normtime grid  [1/s/s].
        """
        En = np.asarray(En)
        d2NdEdt = np.asarray(d2NdEdt)

        # Vectorised linear interpolation of all N_temit columns at once.
        # Find the left-neighbour index for each En_det point in En.
        idx = np.searchsorted(En, self.En_det, side="right") - 1
        idx = np.clip(idx, 0, len(En) - 2)  # shape (N_td,)

        dE = En[idx + 1] - En[idx]
        dE = np.where(dE > 0, dE, 1.0)  # guard against duplicate knots
        t_w = (self.En_det - En[idx]) / dE  # linear weight in [0,1]
        t_w = np.clip(t_w, 0.0, 1.0)

        # Broadcast: (N_td,) x (N_temit,) -> (N_td, N_temit)
        d2NdEdt_interp = (1.0 - t_w)[:, None] * d2NdEdt[idx, :] + t_w[:, None] * d2NdEdt[idx + 1, :]

        # Zero out points outside the supplied energy range
        out_of_range = (self.En_det < En[0]) | (self.En_det > En[-1])
        d2NdEdt_interp[out_of_range, :] = 0.0

        return d2NdEdt_interp * self.dEdt[:, None]  # (N_td, N_temit)

    def _apply_emission_time_shift(self, RS, temit):
        """Apply the emission-time shift W implicitly and integrate over
        emission time.

        Each emission-time bin k contributes to the output via three steps:

        1. **Fractional shift** — column RS[:, k] is shifted by
           ``t_emit[k] / dt_td`` bins using an integer zero-filling roll
           plus sub-bin linear interpolation between the floor and ceil
           rolls.  Zero-filling (not wrap-around) drops out-of-window
           contributions silently.

        2. **Top-hat spread** — the shifted column is convolved with a
           normalised top-hat of width ``round(dt_emit[k] / dt_td)`` bins
           via ``uniform_filter1d`` (sum-preserving, O(N_td) regardless of
           spread width).  This correctly handles emission bins that span
           many detector time bins.

        3. **Integration weight** — multiply by ``dt_emit[k]`` (seconds) to
           integrate d²N/dE dt_emit over the emission-time axis.

        Parameters
        ----------
        RS : ndarray, shape (N_td, N_temit)
            Signal matrix after (optional) IRF application.
        temit : ndarray, shape (N_temit,)
            Emission time bin centres (s).

        Returns
        -------
        signal : ndarray, shape (N_td,)
        """
        td = self.detector_time  # (N_td,)  uniform by construction
        N_td = len(td)
        dt_td = td[1] - td[0]  # uniform detector time bin width (s)

        # Trapezoidal bin widths for integration over t_emit
        dt_emit = np.gradient(temit)  # (N_temit,)

        signal = np.zeros(N_td)

        for k in range(len(temit)):
            col = RS[:, k]

            # ----------------------------------------------------------
            # Step 1: fractional shift
            # Decompose t_emit[k]/dt_td into integer + sub-bin fraction.
            # ----------------------------------------------------------
            shift_bins = temit[k] / dt_td
            n_lo = int(np.floor(shift_bins))
            f = shift_bins - n_lo  # sub-bin fraction in [0, 1)

            shifted = (1.0 - f) * _roll_zero(col, n_lo) + f * _roll_zero(col, n_lo + 1)

            # ----------------------------------------------------------
            # Step 2: top-hat spread over dt_emit[k]
            # uniform_filter1d is sum-preserving and handles any width.
            # ----------------------------------------------------------
            n_spread = max(1, round(dt_emit[k] / dt_td))
            spread = uniform_filter1d(shifted, size=n_spread, mode="constant", cval=0.0)

            # ----------------------------------------------------------
            # Step 3: integrate over emission time
            # ----------------------------------------------------------
            signal += spread * dt_emit[k]

        return signal

    def get_time_resolved_signal(self, En, d2NdEdt, temit):
        """Full time-resolved forward model: interpolation → sensitivity →
        IRF → emission-time shift.

        Parameters
        ----------
        En : array_like, shape (N_E,)
            Energy bin centres (eV).  Must be sorted ascending.
        d2NdEdt : array_like, shape (N_E, N_temit)
            d²N/dE dt_emit  [1/eV/s].
        temit : array_like, shape (N_temit,)
            Emission time bin centres (s). Must be > 0 and sorted ascending.

        Returns
        -------
        detector_time : ndarray, shape (N_td,)
        detector_normtime : ndarray, shape (N_td,)
        signal : ndarray, shape (N_td,)
        """
        temit = np.asarray(temit)
        assert np.all(temit >= 0), "Emission time bins must be > 0"
        assert np.all(np.diff(temit) > 0), "Emission time bins must be sorted ascending"
        dNdt2d = self.get_time_resolved_dNdt(En, d2NdEdt)  # (N_td, N_temit)
        RS = np.matmul(self.R, self.sens[:, None] * dNdt2d)  # (N_td, N_temit)
        signal = self._apply_emission_time_shift(RS, temit)
        return self.detector_time, self.detector_normtime, signal

    def get_time_resolved_signal_no_IRF(self, En, d2NdEdt, temit):
        """Time-resolved forward model without IRF application.

        Parameters
        ----------
        En : array_like, shape (N_E,)
            Energy bin centres (eV).  Must be sorted ascending.
        d2NdEdt : array_like, shape (N_E, N_temit)
            d²N/dE dt_emit  [1/eV/s].
        temit : array_like, shape (N_temit,)
            Emission time bin centres (s). Must be > 0 and sorted ascending.

        Returns
        -------
        detector_time : ndarray, shape (N_td,)
        detector_normtime : ndarray, shape (N_td,)
        signal : ndarray, shape (N_td,)
        """
        temit = np.asarray(temit)
        assert np.all(temit >= 0), "Emission time bins must be > 0"
        assert np.all(np.diff(temit) > 0), "Emission time bins must be sorted ascending"
        dNdt2d = self.get_time_resolved_dNdt(En, d2NdEdt)  # (N_td, N_temit)
        RS = self.sens[:, None] * dNdt2d  # (N_td, N_temit)
        signal = self._apply_emission_time_shift(RS, temit)
        return self.detector_time, self.detector_normtime, signal
