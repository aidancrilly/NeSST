from NeSST.core import *
from NeSST.utils import *


class DT_fit_function:
    """
    A class which constructs various simple models for the full spectrum in DT for fitting data

    User must provide energy grids (and velocity grids if including ion kinematics)

    One can create a model function from the following list of approximations:

    -- Symmetric areal density
    -- Asymmetric Mode 1 areal density

    The primary spectra are assumed isotropic and with moments defined by a single temperature

    This class doesn't represent the full set of spectrum models which can be produced by NeSST! Just a common few...
    """

    _source_labels = ("DT", "DD", "TT")
    _material_labels = ("D", "T")

    def __init__(self, E_DTspec, E_sspec, vion_arr=None):
        self.E_DTspec = E_DTspec
        self.E_sspec = E_sspec
        print("### Initialising data on energy grids... ###")
        init_DT_scatter(E_sspec, E_sspec)
        if vion_arr is not None:
            self.ion_kinematics = True
            self.vion_arr = vion_arr
            print("### Initialising scattering matrices on ion velocity grid... ###")
            init_DT_ionkin_scatter(vion_arr, nT=True, nD=True)
        else:
            self.ion_kinematics = False
        print("### Init Done. ###")

    def set_primary_Tion(self, Tion):
        if Tion < 0.1:
            print("~~ WARNING Low Tion (< 100 eV) ~~")
        self.Tion = Tion

        self.DTmean, _, self.DTvar = DTprimspecmoments(Tion)
        self.DDmean, _, self.DDvar = DDprimspecmoments(Tion)

        Y_DT = 1.0
        Y_DD = yield_from_dt_yield_ratio("dd", Y_DT, Tion)
        Y_TT = yield_from_dt_yield_ratio("tt", Y_DT, Tion)

        self.dNdE_DT = Y_DT * QBrysk(self.E_DTspec, self.DTmean, self.DTvar)
        self.dNdE_DD = Y_DD * QBrysk(self.E_sspec, self.DDmean, self.DDvar)
        self.dNdE_TT = Y_TT * dNdE_TT(self.E_sspec, Tion)

        self.I_DT = interpolate_1d(self.E_DTspec, self.dNdE_DT, fill_value=0.0, bounds_error=False)
        self.I_DD = interpolate_1d(self.E_sspec, self.dNdE_DD)
        self.I_TT = interpolate_1d(self.E_sspec, self.dNdE_TT)
        self._primary_scatter_spectra = {
            "DT": self.I_DT(self.E_sspec),
            "DD": self.dNdE_DD,
            "TT": self.dNdE_TT,
        }

    def _primary_source_weights(self, fT, fD):
        return {
            "DT": 1.0,
            "DD": (fD / fT) * (frac_T_default / frac_D_default),
            "TT": (fT / fD) * (frac_D_default / frac_T_default),
        }

    def _primary_background(self, E, source_weights):
        return source_weights["DD"] * self.I_DD(E) + source_weights["TT"] * self.I_TT(E)

    def _double_scatter_grid(self, first_scatter, fT, fD):
        double_scatter, _ = DT_sym_scatter_spec(first_scatter, frac_D=fD, frac_T=fT)
        return double_scatter

    def _scattered_signal(self, first_scatter, A_1S, fT, fD, include_double_scatter):
        scattered = A_1S * first_scatter
        if include_double_scatter:
            scattered += A_1S**2 * self._double_scatter_grid(first_scatter, fT, fD)
        return scattered

    def _interpolate_scatter(self, spectrum, E):
        interpolator = interpolate_1d(self.E_sspec, spectrum, fill_value=0.0, bounds_error=False)
        return interpolator(E)

    def _init_symmetric_responses(self):
        rhoL_func = lambda x: np.ones_like(x)
        self._symmetric_static = {}
        self._symmetric_ion = {}

        for source_label, primary_spectrum in self._primary_scatter_spectra.items():
            self._symmetric_static[source_label] = {}
            self._symmetric_ion[source_label] = {}
            for material_label in self._material_labels:
                material = mat_dict[material_label]
                material.calc_station_elastic_dNdE(primary_spectrum, rhoL_func)
                material.calc_n2n_dNdE(primary_spectrum, rhoL_func)
                self._symmetric_static[source_label][material_label] = {
                    "elastic": material.elastic_dNdE.copy(),
                    "n2n": material.n2n_dNdE.copy(),
                }
                if self.ion_kinematics:
                    material.scattering_matrix_apply_rhoLfunc(rhoL_func)
                    material.matrix_primspec_int(primary_spectrum)
                    self._symmetric_ion[source_label][material_label] = material.M_prim.copy()

    def _symmetric_first_scatter_grid(self, source_weights, fT, fD, vbar=None, dv=None):
        material_fractions = {"D": fD, "T": fT}
        first_scatter = np.zeros_like(self.E_sspec)

        for material_label, material_fraction in material_fractions.items():
            n2n = sum(
                source_weights[source_label] * self._symmetric_static[source_label][material_label]["n2n"]
                for source_label in self._source_labels
            )
            if self.ion_kinematics:
                material = mat_dict[material_label]
                material.M_prim = sum(
                    source_weights[source_label] * self._symmetric_ion[source_label][material_label]
                    for source_label in self._source_labels
                )
                elastic = material.matrix_interpolate_gaussian(self.E_sspec, vbar, dv)
            else:
                elastic = sum(
                    source_weights[source_label]
                    * self._symmetric_static[source_label][material_label]["elastic"]
                    for source_label in self._source_labels
                )
            first_scatter += material_fraction * (elastic + n2n)

        return first_scatter

    def init_symmetric_model(self, include_double_scatter=False):
        """
        Creates a callable model function for a symmetric areal density distribution.

        The first-scatter source includes DT, DD and TT neutrons. The approximate
        optional double-scatter term uses stationary-ion kernels and the symmetric
        areal-density prescription from the NeSST paper.
        """

        self._init_symmetric_responses()

        if self.ion_kinematics:

            def model(E, rhoL, vbar, dv, fT, fD, Yn):
                """
                Symmetric areal density model with scattering ion velocity distribution with mean and std dev, vbar and dv in m/s
                """
                A_1S = rhoR_2_A1s(rhoL, frac_D=fD, frac_T=fT)
                source_weights = self._primary_source_weights(fT, fD)
                first_scatter = self._symmetric_first_scatter_grid(source_weights, fT, fD, vbar, dv)
                scattered = self._scattered_signal(
                    first_scatter, A_1S, fT, fD, include_double_scatter
                )
                return Yn * (
                    self._interpolate_scatter(scattered, E) + self._primary_background(E, source_weights)
                )

        else:

            def model(E, rhoL, Ts, fT, fD, Yn):
                """
                Symmetric areal density model with scattering temperature Ts, in keV
                """
                A_1S = rhoR_2_A1s(rhoL, frac_D=fD, frac_T=fT)
                source_weights = self._primary_source_weights(fT, fD)
                first_scatter = self._symmetric_first_scatter_grid(source_weights, fT, fD)
                scattered = self._scattered_signal(
                    first_scatter, A_1S, fT, fD, include_double_scatter
                )
                return Yn * (
                    self._interpolate_scatter(scattered, E) + self._primary_background(E, source_weights)
                )

        self.model = model

    def _init_modeone_responses(self, P1_arr):
        self._modeone_n2n = {}
        self._modeone_static = {}
        self._modeone_ion = {}

        for source_label, primary_spectrum in self._primary_scatter_spectra.items():
            self._modeone_n2n[source_label] = {}
            self._modeone_static[source_label] = {}
            self._modeone_ion[source_label] = {}
            for material_label in self._material_labels:
                material = mat_dict[material_label]
                elastic_modeone = np.trapezoid(
                    material.elastic_dNdEdmu[:, :, None]
                    * (1.0 + P1_arr[None, None, :] * material.elastic_mu0[:, :, None])
                    * primary_spectrum[None, :, None],
                    self.E_sspec,
                    axis=1,
                )
                self._modeone_static[source_label][material_label] = interpolate_2d(
                    self.E_sspec, P1_arr, elastic_modeone, bounds_error=False
                )
                n2n_rgrid_IE = np.trapezoid(
                    material.n2n_ddx.rgrid * primary_spectrum[:, None, None], self.E_sspec, axis=0
                )
                n2n_modeone = np.trapezoid(
                    n2n_rgrid_IE[:, :, None]
                    * (1.0 + P1_arr[None, None, :] * material.n2n_mu[:, None, None]),
                    material.n2n_mu,
                    axis=0,
                )
                self._modeone_n2n[source_label][material_label] = interpolate_2d(
                    self.E_sspec, P1_arr, n2n_modeone, bounds_error=False
                )

                if self.ion_kinematics:
                    M_modeone = np.trapezoid(
                        (1.0 + P1_arr[None, None, None, :] * material.full_scattering_mu[:, :, :, None])
                        * material.full_scattering_M[:, :, :, None]
                        * primary_spectrum[None, None, :, None],
                        self.E_sspec,
                        axis=2,
                    )
                    self._modeone_ion[source_label][material_label] = interpolate_1d(
                        P1_arr, M_modeone, axis=-1, bounds_error=False
                    )

    def _modeone_first_scatter_grid(self, source_weights, fT, fD, P1, vbar=None, dv=None):
        material_fractions = {"D": fD, "T": fT}
        first_scatter = np.zeros_like(self.E_sspec)

        for material_label, material_fraction in material_fractions.items():
            if self.ion_kinematics:
                material = mat_dict[material_label]
                material.M_prim = sum(
                    source_weights[source_label] * self._modeone_ion[source_label][material_label](P1)
                    for source_label in self._source_labels
                )
                elastic = material.matrix_interpolate_gaussian(self.E_sspec, vbar, dv)
            else:
                elastic = sum(
                    source_weights[source_label]
                    * self._modeone_static[source_label][material_label](self.E_sspec, P1)
                    for source_label in self._source_labels
                )
            n2n = sum(
                source_weights[source_label]
                * self._modeone_n2n[source_label][material_label](self.E_sspec, P1)
                for source_label in self._source_labels
            )
            first_scatter += material_fraction * (elastic + n2n)

        return first_scatter

    def init_modeone_model(self, P1_arr, include_double_scatter=False):
        """
        Creates a callable model function for a mode 1 asymmetric areal density distribution.

        Double scattering is calculated by applying the symmetric stationary-ion
        kernel to the mode-1 first-scatter signal.
        """

        self.P1_arr = P1_arr
        self._init_modeone_responses(P1_arr)

        if self.ion_kinematics:

            def model(E, rhoL, P1, vbar, dv, fT, fD, Yn):
                A_1S = rhoR_2_A1s(rhoL, frac_D=fD, frac_T=fT)
                source_weights = self._primary_source_weights(fT, fD)
                first_scatter = self._modeone_first_scatter_grid(
                    source_weights, fT, fD, P1, vbar, dv
                )
                scattered = self._scattered_signal(
                    first_scatter, A_1S, fT, fD, include_double_scatter
                )
                return Yn * (
                    self._interpolate_scatter(scattered, E) + self._primary_background(E, source_weights)
                )

        else:

            def model(E, rhoL, P1, Ts, fT, fD, Yn):
                A_1S = rhoR_2_A1s(rhoL, frac_D=fD, frac_T=fT)
                source_weights = self._primary_source_weights(fT, fD)
                first_scatter = self._modeone_first_scatter_grid(source_weights, fT, fD, P1)
                scattered = self._scattered_signal(
                    first_scatter, A_1S, fT, fD, include_double_scatter
                )
                return Yn * (
                    self._interpolate_scatter(scattered, E) + self._primary_background(E, source_weights)
                )

        self.model = model
