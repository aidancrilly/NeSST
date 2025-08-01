import multiprocessing
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import quad
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


def get_power_law_NLO(p, Enorm=E0_DT):
    A = mat_dict["H"].sigma(Enorm) * Enorm**p

    def power_law_NLO(E):
        return mat_dict["H"].sigma(E) * E**p / A

    return power_law_NLO


def get_Verbinski_NLO(Enorm=E0_DT):
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
    V_E, V_L = np.loadtxt(data_dir + "VerbinskiLproton.csv", delimiter=",", unpack=True)
    cumulative_L = cumtrapz(y=np.insert(V_L, 0, 0.0), x=np.insert(V_E, 0, 0.0))
    L_integral = interpolate_1d(np.insert(V_E, 0, 0.0) * 1e6, np.insert(cumulative_L, 0, 0.0), method="cubic")

    def Verbinski_NLO(E):
        return mat_dict["H"].sigma(E) * L_integral(E) / E

    A = Verbinski_NLO(Enorm)

    def norm_Verbinski_NLO(E):
        return Verbinski_NLO(E) / A

    return norm_Verbinski_NLO


def get_BirksBetheBloch_NLO(akB, Enorm=E0_DT):
    r"""
    akB in eV

    Combining:
    kB in m/eV
    a in eV^2/m
    Bethe-Bloch formula for stopping power:
        dE/dx = a/E

    Birks relation for light response:
        dL/dx \propto (dEdx)/(1+kB (dE/dx))

    Using equation (3), (7) from

    Qi Tang, Zifeng Song, Pinyang Liu, Bo Yu, Jiamin Yang,
    Calibration of the sensitivity of the bibenzyl-based scintillation detector to 1–5 MeV neutrons,
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment,
    Volume 1068,
    2024,
    169779,
    ISSN 0168-9002,
    https://doi.org/10.1016/j.nima.2024.169779.
    """

    def BirksBetheBloch_NLO(E):
        L_integral = 0.5 * E**2 - akB * E - akB * (akB + E) * np.log(1.0 + E / akB)
        return mat_dict["H"].sigma(E) * L_integral / E

    A = BirksBetheBloch_NLO(Enorm)

    def norm_BirksBetheBloch_NLO(E):
        return BirksBetheBloch_NLO(E) / A

    return norm_BirksBetheBloch_NLO


def get_BirksBethe_NLO(akB, excitation_energy, mp=sc.m_p, Enorm=E0_DT):
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

    Using equation (3), (7) from

    Qi Tang, Zifeng Song, Pinyang Liu, Bo Yu, Jiamin Yang,
    Calibration of the sensitivity of the bibenzyl-based scintillation detector to 1–5 MeV neutrons,
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment,
    Volume 1068,
    2024,
    169779,
    ISSN 0168-9002,
    https://doi.org/10.1016/j.nima.2024.169779.
    """
    Istar = excitation_energy * mp / sc.m_e / 4.0

    def kB_dEdx(Ep):
        Ep_lim = max([Ep, np.e * Istar])
        return akB / Ep_lim * np.log(Ep_lim / Istar)

    def dLdE(Ep):
        return 1.0 / (1.0 + kB_dEdx(Ep))

    def L(Ep):
        return quad(dLdE, 0.0, Ep)[0]

    def L_integral_scalar(En):
        return quad(L, 0.0, En)[0]

    L_integral = np.vectorize(L_integral_scalar)

    def BirksBethe_NLO(E):
        return mat_dict["H"].sigma(E) * L_integral(E) / E

    A = BirksBethe_NLO(Enorm)

    def norm_BirksBethe_NLO(E):
        return BirksBethe_NLO(E) / A

    return norm_BirksBethe_NLO, kB_dEdx, L


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


class Scintillator1DMonteCarlo:
    """

    A 1D Monte Carlo simulator of the temporal response of a scintillator

    """

    def __init__(self, num_workers=None, E_threshold=1e2):
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.E_threshold = E_threshold

    def sig_p(self, E):
        """
        Taken from "Radiation detection and measurement" by G Knoll
        """
        return (4.83 / np.sqrt(E / 1e6) - 0.578) * 1e-28

    def E_2_v(self, E):
        """
        Energy in eV to speed in m/s
        """
        E_J = E * sc.e
        return np.sqrt(2 * E_J / sc.m_n)

    def simulate_path(self, args):
        """
        Simulate the path of a single neutron entering a 1D scintillator of length l and proton density ni (m and m^-3 respectively)

        Returns list of interaction times and energy deposited at that time
        """
        Ein, ni, l = args
        x = 0
        dir = +1
        E = Ein
        v = self.E_2_v(E)
        sig = self.sig_p(E)
        in_scintillator = True
        t = 0

        ts = []
        Es = []
        while in_scintillator:
            path_to_exit = (l - x) * (dir + 1) / 2.0 + x * (1 - dir) / 2.0
            path = -np.log(np.random.rand()) / (ni * sig)
            if path < path_to_exit:
                t += path / v
                x += dir * path
                muc = 2 * np.random.rand() - 1.0
                Enext = E * (2.0 + 2.0 * muc) / 4.0
                ts.append(t)
                Es.append(E - Enext)

                mu = np.sqrt(Enext / E)
                E = Enext
                v = self.E_2_v(E)
                sig = self.sig_p(E)
            else:
                in_scintillator = False

            if E < self.E_threshold:
                in_scintillator = False

        return np.array(ts), np.array(Es)

    def __call__(self, E_beam, ni_scin, l_scin, Nsimulated_paths, Nbins=1000):
        """
        Run the Monte Carlo simulation for a single neutron beam energy.

        Parameters:
        -----------
        E_beam : float
            Initial energy of the neutron beam in eV.
        ni_scin : float
            Number density of protons in the scintillator in (1/m^3)
        l_scin : float
            Thickness of the scintillator in meters.
        Nsimulated_paths : int
            Number of neutron paths to simulate.
        Nbins : int
            Number of histogram bins for time distribution, default = 1000

        Returns:
        --------
        hist : np.ndarray
            Normalized histogram of energy depositions over time (density=True).
        tbins : np.ndarray
            Bin edges in nanoseconds for the time histogram.
        """
        v_beam = self.E_2_v(E_beam)
        tscale = l_scin / v_beam
        tbins = np.linspace(0.0, 5.0 * tscale, Nbins + 1)

        args = [(E_beam, ni_scin, l_scin)] * Nsimulated_paths

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.map(self.simulate_path, args)

        list_of_ts, list_of_Es = zip(*results, strict=False)
        ts = np.concatenate(list_of_ts)
        Es = np.concatenate(list_of_Es)

        hist, _ = np.histogram(ts, weights=Es, bins=tbins, density=True)
        return hist, tbins


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


def inversegaussian_nIRF(scint_thickness, ni_scin = 1e29):
    def sig_p(E):
        return (4.83 / np.sqrt(E / 1e6) - 0.578) * 1e-28

    if scint_thickness != 10e-2 or ni_scin != 1e29:
        raise NotImplementedError("Current inverse gaussian nIRF fit coefficients for 1e29 1/m^3 and 10cm only!")

    def IGtail_nIRF_bestfit_coeffs(En, scint_thickness, ni_scin):
        """
        Best fit coefficients as a function of neutron energy

        NB these are specific to a scintillator geometry (10cm, 1e29 p/m^3)
        For different geometry you must re-perform Scintillator1DMonteCarlo sims

        Analysis performed by A. Crilly, 01/08/2025
        """
        E_MeV = En / 1e6
        f = -0.0054769 * E_MeV + 0.50115357
        A = (0.30110498 - 0.00491061 * E_MeV) * (1 - np.exp(-E_MeV / 2.0801267))
        mu_inverse_ns = -0.3371536 + E_MeV**0.59130151
        lamb_inverse_ns = 0.18257606 * E_MeV + 2.62378105
        mu = mu_inverse_ns*1e9
        lamb = lamb_inverse_ns*1e9
        return f, A, mu, lamb

    def base(t_detected, En):
        vn = Ekin_2_beta(En, Mn) * c
        t_transit = scint_thickness / vn

        tt_d, tt_a = np.meshgrid(t_detected, t_detected, indexing="ij")
        _, tt_t = np.meshgrid(t_detected, t_transit, indexing="ij")

        

        f, A, mu, lamb = IGtail_nIRF_bestfit_coeffs(En, scint_thickness, ni_scin)

        top_hat = np.eye(t_detected.size) + np.heaviside(tt_d - tt_a, 0.0) - np.heaviside(tt_d - (tt_a + tt_t), 1.0)
        exp_E_arg = f * vn * ni_scin * sig_p(En)
        main_response = np.exp(- (tt_d - tt_a) * exp_E_arg[None, :]) * top_hat

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
    # Fallback to the usual top-hat if the caller doesn’t supply one
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
