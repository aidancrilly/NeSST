import dress

def DRESS_DT_spec_single_T(T,n_samples,bins):
    # Conversion to keV units of DRESS
    T_keV = T/1e3
    bins /= 1e3

    reaction = dress.reactions.DTNHe4Reaction()
    spec_calc = dress.SpectrumCalculator(reaction, n_samples=n_samples)

    dist_a = dress.dists.MaxwellianDistribution(T_keV, spec_calc.reactant_a.particle)
    dist_b = dress.dists.MaxwellianDistribution(T_keV, spec_calc.reactant_b.particle)

    spec_calc.reactant_a.v = dist_a.sample(spec_calc.n_samples)
    spec_calc.reactant_b.v = dist_b.sample(spec_calc.n_samples)

    spec = spec_calc(bins=bins,normalize=True)

    # Ensure unit area for eV
    bins_w = 1e3*(bins[1:]-bins[:-1])

    return spec/bins_w