import NeSST as nst
import numpy as np
import pytest
import pytest_cases


@pytest_cases.fixture(
    params=[
        "delta",
        "decaying_gaussian",
        "gated_decaying_gaussian_kernel",
        "double_decay_gaussian_kernel",
        "t_gaussian_kernel",
    ]
)
def kernel_fn(request):
    if request.param == "delta":
        kernel = nst.time_of_flight.delta_kernel()
    elif request.param == "decaying_gaussian":
        kernel = nst.time_of_flight.decaying_gaussian_kernel(FWHM=2.5e-9, tau=2e-9)
    elif request.param == "gated_decaying_gaussian_kernel":
        kernel = nst.time_of_flight.gated_decaying_gaussian_kernel(
            sig=1.0e-9, tau=2e-9, shift_t=1.0e-9, sig_turnon=0.5e-9
        )
    elif request.param == "double_decay_gaussian_kernel":
        kernel = nst.time_of_flight.double_decay_gaussian_kernel(FWHM=2.5e-9, taus=[2e-9, 10e-9], frac=0.5)
    elif request.param == "t_gaussian_kernel":
        kernel = nst.time_of_flight.t_gaussian_kernel(FWHM=2.5e-9, peak_pos=2e-9)

    return kernel


def test_IRF_normalisation(kernel_fn):
    """

    For uniform sensitivity, the transform to ntof space should be conservative

    """
    nToF_distance = 20.0  # m
    flat_sens = nst.time_of_flight.get_unity_sensitivity()
    total_IRF = nst.time_of_flight.make_transit_time_IRF(thickness=0.1, kernel_fn=kernel_fn)

    E_ntof = np.linspace(1.0e6, 10.0e6, 500)
    DDmean, _, DDvar = nst.DDprimspecmoments(Tion=5.0e3)
    dNdE = nst.QBrysk(E_ntof, DDmean, DDvar)

    test_20m_nToF = nst.time_of_flight.nToF(nToF_distance, flat_sens, total_IRF)
    t_det, normt_det, signal = test_20m_nToF.get_signal(E_ntof, dNdE)

    integral_signal = np.trapezoid(y=signal, x=normt_det)

    assert np.isclose(integral_signal, 1.0, rtol=1e-3)


def test_inversegaussian_nIRF_normalisation():
    """
    For uniform sensitivity, the inversegaussian_nIRF response matrix rows should sum to 1
    (i.e. the response is normalised).
    """
    nToF_distance = 20.0  # m
    flat_sens = nst.time_of_flight.get_unity_sensitivity()
    total_IRF = nst.time_of_flight.inversegaussian_nIRF(scint_thickness=10e-2)

    E_ntof = np.linspace(1.0e6, 10.0e6, 500)
    DDmean, _, DDvar = nst.DDprimspecmoments(Tion=5.0e3)
    dNdE = nst.QBrysk(E_ntof, DDmean, DDvar)

    test_20m_nToF = nst.time_of_flight.nToF(nToF_distance, flat_sens, total_IRF)
    t_det, normt_det, signal = test_20m_nToF.get_signal(E_ntof, dNdE)

    integral_signal = np.trapezoid(y=signal, x=normt_det)

    assert np.isclose(integral_signal, 1.0, rtol=1e-2)


@pytest_cases.fixture(
    params=[
        "get_power_law_NLO",
        "get_Verbinski_NLO",
        "get_BirksBetheBloch_NLO",
        "get_BirksBethe_NLO",
        "get_CraunSmithBethe_NLO",
    ]
)
def NLO_fn(request):
    if request.param == "get_power_law_NLO":
        NLO = nst.time_of_flight.get_power_law_NLO(p=1.5)
    elif request.param == "get_Verbinski_NLO":
        NLO = nst.time_of_flight.get_Verbinski_NLO()
    elif request.param == "get_BirksBetheBloch_NLO":
        NLO = nst.time_of_flight.get_BirksBetheBloch_NLO(akB=1e6)
    elif request.param == "get_BirksBethe_NLO":
        NLO = nst.time_of_flight.get_BirksBethe_NLO(akB=1e6, excitation_energy=1e2)
    elif request.param == "get_CraunSmithBethe_NLO":
        NLO = nst.time_of_flight.get_CraunSmithBethe_NLO(C=0.1, akB=1e6, excitation_energy=1e2)

    return NLO


def test_NLO_monotonic(NLO_fn):
    """
    Test that the NLO functions are monotonic increasing
    """
    E = np.linspace(1e3, 20e6, 500)  # eV
    S = NLO_fn(E)

    assert np.all(np.diff(S) > 0)

N_MC = 1_000_000
_MC_SEED = 42


def _mc_arrival_times(distance, Earr, d2NdEdt, tarr, rng):
    # Joint probability mass on the grid (not normalised yet)
    dE = np.gradient(Earr)           # (N_E,)
    dt = np.gradient(tarr)           # (N_temit,)
    mass2d = d2NdEdt * dE[:, None] * dt[None, :]   # (N_E, N_temit)
    mass2d = np.maximum(mass2d, 0.0)
    total = mass2d.sum()
    flat_prob = mass2d.ravel() / total

    # Sample flat indices
    flat_idx = rng.choice(len(flat_prob), size=N_MC, p=flat_prob)
    ie, it = np.unravel_index(flat_idx, mass2d.shape)

    # Jitter within each cell (uniform sub-cell noise)
    E_samples = Earr[ie] + (rng.random(N_MC) - 0.5) * dE[ie]
    t_emit_samples = tarr[it] + (rng.random(N_MC) - 0.5) * dt[it]

    # Relativistic transit time
    beta = nst.col.Ekin_2_beta(E_samples, nst.Mn)
    t_transit = distance / (beta * nst.c)

    return t_transit + t_emit_samples


@pytest.mark.parametrize("distance", [0.5, 5.0, 20.0])
def test_time_resolved_no_IRF_vs_MC(distance):
    Tion = 5.0e3   # eV
    BW   = 1.0e-9  # s  (burn-width FWHM)
    sigma_bw = BW / 2.355

    def burn_history(t):
        return np.exp(-0.5 * ((t - 3 * sigma_bw) / sigma_bw) ** 2) / (sigma_bw * np.sqrt(2 * np.pi))

    DTmean, _, DTvar = nst.DTprimspecmoments(Tion)

    tarr = np.linspace(0, 10e-9, 200)
    Earr = np.linspace(13.0e6, 15.0e6, 200)

    d2NdEdt = nst.QBrysk(Earr, DTmean, DTvar)[:, None] * burn_history(tarr)[None, :]

    # ---------------------------------------------------------------
    # NeSST forward model (no IRF, unity sensitivity)
    # The normtime window must be wide enough to contain the full
    # time-resolved signal: arrival of slowest neutron + last emission.
    # ---------------------------------------------------------------
    flat_sens = nst.time_of_flight.get_unity_sensitivity()
    delta_irf  = nst.time_of_flight.make_transit_time_IRF(
        thickness=0.0,
        kernel_fn=lambda t: np.array([1.0]),
    )
    # slowest neutron normtime (at Earr[0]) plus emission window padded
    normtime_lo = nst.col.Ekin_2_beta(Earr[-1], nst.Mn) ** -1 * 0.95
    normtime_hi = (
        nst.col.Ekin_2_beta(Earr[0], nst.Mn) ** -1
        + tarr[-1] * nst.c / distance
    ) * 1.05

    det = nst.time_of_flight.nToF(
        distance,
        flat_sens,
        delta_irf,
        normtime_start=normtime_lo,
        normtime_end=normtime_hi,
        normtime_N=2048,
    )

    t_det, _, nesst_signal = det.get_time_resolved_signal_no_IRF(Earr, d2NdEdt, tarr)

    # ---------------------------------------------------------------
    # Monte Carlo reference
    # ---------------------------------------------------------------
    rng = np.random.default_rng(_MC_SEED)
    t_arrive_mc = _mc_arrival_times(distance, Earr, d2NdEdt, tarr, rng)

    # Histogram onto the same detector_time bin edges
    dt_td = t_det[1] - t_det[0]
    bin_edges = np.concatenate([t_det - 0.5 * dt_td, [t_det[-1] + 0.5 * dt_td]])
    mc_hist, _ = np.histogram(t_arrive_mc, bins=bin_edges)
    mc_hist = mc_hist.astype(float)

    # ---------------------------------------------------------------
    # Normalise both to unit sum (comparison is shape only)
    # ---------------------------------------------------------------
    nesst_norm = nesst_signal / nesst_signal.sum()
    mc_norm    = mc_hist / mc_hist.sum()

    # Tolerance: 5 * Poisson sigma at the peak MC bin
    peak_frac   = mc_norm.max()
    peak_counts = peak_frac * N_MC
    atol = 5.0 * np.sqrt(peak_counts) / N_MC   # 5-sigma at the peak bin

    assert np.allclose(nesst_norm, mc_norm, atol=atol), (
        f"distance={distance} m: max |NeSST - MC| = "
        f"{np.abs(nesst_norm - mc_norm).max():.3e}  (tol={atol:.3e})"
    )
