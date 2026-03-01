import numpy as np
import numpy.typing as npt

try:
    import dress

    dress_available = True
except ImportError:
    dress_available = False


def _check_dress():
    if not dress_available:
        raise ImportError("pydress is required for DRESS interface functions. Install it with: pip install pydress")


def Ecentres_to_edges(Ecentres: npt.NDArray) -> npt.NDArray:
    """Convert energy bin centres to edges.

    Args:
        Ecentres (numpy.array): energy bin centres in eV

    Returns:
        numpy.array: energy bin edges in eV
    """
    dE = np.diff(Ecentres)
    Eedges = np.empty(len(Ecentres) + 1)
    Eedges[1:-1] = 0.5 * (Ecentres[:-1] + Ecentres[1:])
    Eedges[0] = Ecentres[0] - 0.5 * dE[0]
    Eedges[-1] = Ecentres[-1] + 0.5 * dE[-1]
    return Eedges, Eedges[1:] - Eedges[:-1]


def DRESS_DT_spec(T_D: float, T_T: float, n_samples: int, bins: npt.NDArray) -> npt.NDArray:
    """Computes DT primary neutron spectrum using DRESS with separate D and T temperatures.

    Args:
        T_D (float): temperature of deuterons in eV
        T_T (float): temperature of tritons in eV
        n_samples (int): number of Monte Carlo samples
        bins (numpy.array): energy bin edges in eV

    Returns:
        numpy.array: normalised DT spectrum (1/eV) on the bin centres
    """
    _check_dress()
    T_D_keV = T_D / 1e3
    T_T_keV = T_T / 1e3
    bins_keV = bins / 1e3

    reaction = dress.reactions.DTNHe4Reaction()
    spec_calc = dress.SpectrumCalculator(reaction, n_samples=n_samples)

    dist_a = dress.dists.MaxwellianDistribution(T_D_keV, spec_calc.reactant_a.particle)
    dist_b = dress.dists.MaxwellianDistribution(T_T_keV, spec_calc.reactant_b.particle)

    spec_calc.reactant_a.v = dist_a.sample(spec_calc.n_samples)
    spec_calc.reactant_b.v = dist_b.sample(spec_calc.n_samples)

    spec = spec_calc(bins=bins_keV, normalize=True)

    # Convert from unit-sum to unit-area (1/eV)
    bins_w = bins[1:] - bins[:-1]

    return spec / bins_w


def DRESS_DT_spec_single_T(T: float, n_samples: int, bins: npt.NDArray) -> npt.NDArray:
    """Computes DT primary neutron spectrum using DRESS with a single ion temperature.

    Wrapper around DRESS_DT_spec for the case where D and T share the same temperature.

    Args:
        T (float): ion temperature in eV
        n_samples (int): number of Monte Carlo samples
        bins (numpy.array): energy bin edges in eV

    Returns:
        numpy.array: normalised DT spectrum (1/eV) on the bin centres
    """
    return DRESS_DT_spec(T, T, n_samples, bins)


def DRESS_DD_spec(T: float, n_samples: int, bins: npt.NDArray) -> npt.NDArray:
    """Computes DD primary neutron spectrum using DRESS.

    Args:
        T (float): temperature of deuterons in eV
        n_samples (int): number of Monte Carlo samples
        bins (numpy.array): energy bin edges in eV

    Returns:
        numpy.array: normalised DD spectrum (1/eV) on the bin centres
    """
    _check_dress()
    T_keV = T / 1e3
    bins_keV = bins / 1e3

    reaction = dress.reactions.DDNHe3Reaction()
    spec_calc = dress.SpectrumCalculator(reaction, n_samples=n_samples)

    dist_a = dress.dists.MaxwellianDistribution(T_keV, spec_calc.reactant_a.particle)
    dist_b = dress.dists.MaxwellianDistribution(T_keV, spec_calc.reactant_b.particle)

    spec_calc.reactant_a.v = dist_a.sample(spec_calc.n_samples)
    spec_calc.reactant_b.v = dist_b.sample(spec_calc.n_samples)

    spec = spec_calc(bins=bins_keV, normalize=True)

    # Convert from unit-sum to unit-area (1/eV)
    bins_w = bins[1:] - bins[:-1]

    return spec / bins_w
