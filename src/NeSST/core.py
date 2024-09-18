### ###################################### ###
### NeSST - Neutron Scattered Spectra Tool ###
### ###################################### ###

# Standard libraries
import typing
import numpy.typing as npt
import warnings
import numpy as np
from glob import glob
from os.path import basename
# NeSST libraries
from NeSST.constants import *
import NeSST.collisions as col
import NeSST.spectral_model as sm

# Global variable defaults
col.classical_collisions = False
# Atomic fraction of D and T in scattering medium and source
frac_D_default = 0.5
frac_T_default = 0.5

# Units are SI
# Energies, temperatures eV
# Velocities m/s

############################
# Default material load in #
############################
def initialise_material_data(label):
    # Aliases
    if(label == 'H'):
        json = 'H1.json'
    elif(label == 'D'):
        json = 'H2.json'
    elif(label == 'T'):
        json = 'H3.json'
    # Parse name
    else:
        mat_jsons = glob(data_dir+'*.json')
        mats      = [basename(f).split('.')[0] for f in mat_jsons]
        if(label in mats):
            json = label+'.json'
        else:
            print("Material label '"+label+"' not recognised")
            return
    mat_data = sm.material_data(label,json)
    mat_dict[label] = mat_data
    available_materials.append(label)

for mat in default_mat_list:
    initialise_material_data(mat)

##########################################
# Primary spectral shapes & reactivities #
##########################################

# Gaussian "Brysk"
def QBrysk(Ein : npt.NDArray, mean : float, variance : float) -> npt.NDArray:
    """Calculates the primary spectrum with a Brysk shape i.e. Gaussian
    Args:
        Ein (numpy.array) : array of energy values on which to compute spectrum
        mean (float) : mean of spectrum
        variance (float): variance of spectraum

    Returns:
        numpy.array : array with Gaussian spectrum on array Ein
    
    """
    spec = np.exp(-(Ein-mean)**2/2.0/variance)/np.sqrt(2*np.pi*variance)
    return spec

# Ballabio
def QBallabio(Ein : npt.NDArray, mean : float, variance : float) -> npt.NDArray:
    """Calculates the primary spectrum with a Ballabio shape i.e. modified Gaussian
    See equations 44 - 46 of Ballabio et al.

    Args:
        Ein (numpy.array) : array of energy values on which to compute spectrum
        mean (float) : mean of spectrum
        variance (float): variance of spectraum

    Returns:
        numpy.array : array with modified Gaussian spectrum on array Ein
    
    """
    common_factor = 1-1.5*variance/mean**2
    Ebar = mean*np.sqrt(common_factor)
    sig2 = 4.0/3.0*mean**2*(np.sqrt(common_factor)-common_factor)
    norm = np.sqrt(2*np.pi*variance)
    spec = np.exp(-2.0*Ebar*(np.sqrt(Ein)-np.sqrt(Ebar))**2/sig2)/norm
    return spec

# TT spectral shape
def dNdE_TT(E : npt.NDArray, Tion : float) -> npt.NDArray:
    """Calculates the TT primary spectrum with Doppler broadening effect as 
    calculated in Appelbe et al. HEDP 2016

    Args:
        E (numpy.array) : array of energy values on which to compute spectrum (eV)
        Tion (float) : the temperature of the ions in eV

    Returns:
        numpy.array : array with normalised TT spectral shape at energies E
    
    """
    return sm.TT_2dinterp(E,Tion)

def yield_from_dt_yield_ratio(reaction : str, dt_yield : float, Tion : float,
                              frac_D: float = frac_D_default, frac_T: float = frac_T_default) -> float:
    """ Reactivity ratio to predict yield from the DT yield assuming same volume and burn time
        rate_ij = (f_{i}*f_{j}*sigmav_{i,j}(T))/(1+delta_{i,j})  # dN/dVdt
        yield_ij = (rate_ij/rate_dt)*yield_dt

        Uses default models for reactivities
        Note that the TT reaction produces two neutrons.

    Args:
        reaction (string) : what reaction 'dd' or 'tt'
        dt_yield (float) : the yield of DT neutrons, used to scale DD and TT yields
        Tion (float): the temperature of the ions in eV
        frac_D (float) : fraction of D in fuel
        frac_T (float) : fraction of T in fuel

    Raises:
        ValueError: if the Tion is below 0 then a ValueError is raised

    Returns:
        float : Yield of requested reaction

    """

    if Tion < 0:
        raise ValueError("Tion (temperature of the ions) can not be below 0")
    
    if sum([frac_D, frac_T]) != 1.:
        msg = (f'The frac_D ({frac_D_default}) and frac_T ({frac_T_default}) '
               'arguments on the yield_from_dt_yield_ration method do not sum to 1.')
        warnings.warn(msg)

    if reaction == 'tt':
        ratio = (0.5*frac_T*sm.reac_TT(Tion))/(frac_D*sm.reac_DT(Tion))
        ratio = 2.* ratio # Two neutrons are generated for each reaction
    elif reaction == 'dd':
        ratio = (0.5*frac_D*sm.reac_DD(Tion))/(frac_T*sm.reac_DT(Tion))
    else:
        raise ValueError(f'reaction should be either "dd" or "tt" not {reaction}')

    return ratio*dt_yield

def yields_normalised(Tion : float, frac_D: float = frac_D_default, frac_T: float = frac_T_default
                        )  -> typing.Tuple[float, float, float]:
    """ Assuming same volume and burn time, find fractional yields of DT, DD and TT respectively

        Uses default models for reactivities
        Note that the TT reaction produces two neutrons.

    Args:
        Tion (float): the temperature of the ions in eV
        frac_D (float) : fraction of D in fuel
        frac_T (float) : fraction of T in fuel

    Raises:
        ValueError: if the Tion is below 0 then a ValueError is raised

    Returns:
        typing.Tuple[float, float, float]: DT, DD and TT yields, normalised sum of unity

    """

    if Tion < 0:
        raise ValueError("Tion (temperature of the ions) can not be below 0")
    
    if sum([frac_D, frac_T]) != 1.:
        msg = (f'The frac_D ({frac_D_default}) and frac_T ({frac_T_default}) '
               'arguments on the yield_from_dt_yield_ration method do not sum to 1.')
        warnings.warn(msg)

    unnormed_dt_yield = frac_D*frac_T*sm.reac_DT(Tion)
    unnormed_dd_yield = 0.5*frac_D*frac_D*sm.reac_DD(Tion)
    unnormed_tt_yield = 2.0*0.5*frac_T*frac_T*sm.reac_TT(Tion)

    tot_yield = unnormed_dt_yield+unnormed_dd_yield+unnormed_tt_yield

    return unnormed_dt_yield/tot_yield,unnormed_dd_yield/tot_yield,unnormed_tt_yield/tot_yield

###############################################################################
# Ballabio fits, see Table III of L. Ballabio et al 1998 Nucl. Fusion 38 1723 #
###############################################################################

# Returns the mean and variance based on Ballabio
def DTprimspecmoments(Tion: float) -> typing.Tuple[float, float, float]:
    """Calculates the mean energy and the variance of the neutron energy
    emitted during DT fusion accounting for temperature of the incident ions.
    Based on Ballabio fits, see Table III of L. Ballabio et al 1998 Nucl.
    Fusion 38 1723

    Args:
        Tion (float): the temperature of the ions in eV

    Raises:
        ValueError: if the Tion is below 0 then a ValueError is raised

    Returns:
        typing.Tuple[float, float, float]: the mean neutron energy, std deviation and variance in eV
    """

    if Tion < 0:
        raise ValueError("Tion (temperature of the ions) can not be below 0")

    # Mean calculation
    a1 = 5.30509
    a2 = 2.4736e-3
    a3 = 1.84
    a4 = 1.3818

    Tion_kev = Tion / 1e3  # Ballabio equation accepts KeV units
    mean_shift = (
        a1 * Tion_kev ** (0.6666666666) / (1.0 + a2 * Tion_kev**a3) + a4 * Tion_kev
    )
    mean_shift *= 1e3  # converting back to eV

    mean = E0_DT + mean_shift

    # Variance calculation
    omega0 = 177.259
    a1 = 5.1068e-4
    a2 = 7.6223e-3
    a3 = 1.78
    a4 = 8.7691e-5

    delta = a1 * Tion_kev ** (0.6666666666) / (1.0 + a2 * Tion_kev**a3) + a4 * Tion_kev

    C = omega0 * (1 + delta)
    FWHM2 = C**2 * Tion_kev
    variance = FWHM2 / (2.3548200450309493) ** 2
    variance *= 1e6  # converting keV^2 back to eV^2
    stddev = np.sqrt(variance)

    return mean, stddev, variance

# Returns the mean and variance based on Ballabio
def DDprimspecmoments(Tion: float) -> typing.Tuple[float, float, float]:
    """Calculates the mean energy and the variance of the neutron energy
    emitted during DD fusion accounting for temperature of the incident ions.
    Based on Ballabio fits, see Table III of L. Ballabio et al 1998 Nucl.
    Fusion 38 1723

    Args:
        Tion (float): the temperature of the ions in eV

    Raises:
        ValueError: if the Tion is below 0 then a ValueError is raised

    Returns:
        typing.Tuple[float, float, float]: the mean neutron energy, std deviation and variance in eV
    """

    if Tion < 0:
        raise ValueError("Tion (temperature of the ions) can not be below 0")

    # Mean calculation
    a1 = 4.69515
    a2 = -0.040729
    a3 = 0.47
    a4 = 0.81844

    Tion_kev = Tion / 1e3  # Ballabio equation accepts KeV units
    mean_shift = (
        a1 * Tion_kev ** (0.6666666666) / (1.0 + a2 * Tion_kev**a3) + a4 * Tion_kev
    )
    mean_shift *= 1e3  # converting back to eV
    mean = E0_DD + mean_shift

    # Variance calculation
    omega0 = 82.542
    a1 = 1.7013e-3
    a2 = 0.16888
    a3 = 0.49
    a4 = 7.9460e-4

    delta = a1 * Tion_kev ** (0.6666666666) / (1.0 + a2 * Tion_kev**a3) + a4 * Tion_kev

    C = omega0 * (1 + delta)
    FWHM2 = C**2 * Tion_kev
    variance = FWHM2 / (2.3548200450309493) ** 2
    variance *= 1e6  # converting keV^2 back to eV^2
    stddev = np.sqrt(variance)

    return mean, stddev, variance

def neutron_velocity_addition(Ek,u):
    return col.velocity_addition_to_Ekin(Ek,Mn,u)

#######################################
# DT scattered spectra initialisation #
#######################################

def init_DT_scatter(Eout : npt.NDArray, Ein : npt.NDArray):
    """Initialise the scattering matrices for D and T materials

    Args:
        Ein (numpy.array): the array on incoming neutron energies
        Eout (numpy.array): the array on outgoing neutron energies
    
    """
    mat_dict['D'].init_energy_grids(Eout,Ein)
    mat_dict['T'].init_energy_grids(Eout,Ein)
    mat_dict['D'].init_station_scatter_matrices()
    mat_dict['T'].init_station_scatter_matrices()

def init_DT_ionkin_scatter(varr : npt.NDArray, nT: bool = False, nD: bool = False):
    """Initialise the scattering matrices including the effect of ion 
    velocities in the kinematics

    N.B. the static ion scattering matrices must already be calculated
    e.g. by calling init_DT_scatter

    Args:
        Ein (numpy.array): the array on incoming neutron energies
        Eout (numpy.array): the array on outgoing neutron energies
    
    """
    if(nT):
        if(mat_dict['T'].Ein is None):
            print("nT - Needed to initialise energy grids - see init_DT_scatter")
        else:
            mat_dict['T'].full_scattering_matrix_create(varr)
    if(nD):
        if(mat_dict['D'].Ein is None):
            print("nD - Needed to initialise energy grids - see init_DT_scatter")
        else:
            mat_dict['D'].full_scattering_matrix_create(varr)

def calc_DT_ionkin_primspec_rhoL_integral(I_E : npt.NDArray, rhoL_func=None, nT: bool = False, nD: bool = False):
    if(nT):
        if(mat_dict['T'].vvec is None):
            print("nT - Needed to initialise velocity grid - see init_DT_ionkin_scatter")
        else:
            if(rhoL_func is not None):
                mat_dict['T'].scattering_matrix_apply_rhoLfunc(rhoL_func)
            mat_dict['T'].matrix_primspec_int(I_E)
    if(nD):
        if(mat_dict['D'].vvec is None):
            print("nD - Needed to initialise velocity grid - see init_DT_ionkin_scatter")
        else:
            if(rhoL_func is not None):
                mat_dict['D'].scattering_matrix_apply_rhoLfunc(rhoL_func)
            mat_dict['D'].matrix_primspec_int(I_E)


###################################
# General material initialisation #
###################################
def init_mat_scatter(Eout : npt.NDArray, Ein : npt.NDArray, mat_label : str):
    """General material version of init_DT_scatter as specified by material label

    N.B. the mat_lable must match those in available_materials_dict

    Args:
        Ein (numpy.array): the array on incoming neutron energies
        Eout (numpy.array): the array on outgoing neutron energies
        mat_label (str) : material label
    
    
    """
    mat = mat_dict[mat_label]
    mat.init_energy_grids(Eout,Ein)
    mat.init_station_scatter_matrices()
    return mat

###########################################
#   Single Evalutation Scattered Spectra  #
###########################################

def DT_sym_scatter_spec(I_E : npt.NDArray
                        ,frac_D: float = frac_D_default, frac_T: float = frac_T_default
                        ) -> typing.Tuple[npt.NDArray,
                             typing.Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]]:
    """Calculates the single scattered neutron spectrum for DT given a 
    primary neutron spectrum of I_E from isotropic areal density

    This requires the scattering matrices to have been pre-calculated

    The primary neutron spectrum, I_E, is assumed to be on the same energy grid as 
    the incoming energy grid used to calculate the scattering matrices

    Args:
        I_E (numpy.array): the neutron spectrum at Ein energies

    Returns:
        Tuple of numpy.arrays: the total scattered spectrum and a tuple of the components 
        (nD,nT,Dn2n,Tn2n)
    
    """
    rhoL_func = lambda x : np.ones_like(x)
    mat_dict['D'].calc_dNdEs(I_E,rhoL_func)
    mat_dict['T'].calc_dNdEs(I_E,rhoL_func)
    nD   = frac_D*mat_dict['D'].elastic_dNdE
    nT   = frac_T*mat_dict['T'].elastic_dNdE
    Dn2n = frac_D*mat_dict['D'].n2n_dNdE
    Tn2n = frac_T*mat_dict['T'].n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

def DT_asym_scatter_spec(I_E : npt.NDArray, rhoL_func : callable,
                         frac_D: float = frac_D_default, frac_T: float = frac_T_default
                         ) -> typing.Tuple[npt.NDArray,
                             typing.Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]]:
    """Calculates the single scattered neutron spectrum for DT given a 
    primary neutron spectrum of I_E from anisotropic areal density

    This requires the scattering matrices to have been pre-calculated

    The primary neutron spectrum, I_E, is assumed to be on the same energy grid as 
    the incoming energy grid used to calculate the scattering matrices

    The areal density function rhoL_func needs to be a callable function with a 
    single argument (cosine[theta])

    Args:
        I_E (numpy.array): the neutron spectrum at Ein energies
        rhoL_func (callable): must be a single argument function f(x), 
        where x e [-1,1] and f(x) e [0,inf] and int f(x) dx = 1
        frac_D (float) : fraction of D in fuel
        frac_T (float) : fraction of T in fuel

    Returns:
        Tuple of numpy.arrays: the total scattered spectrum and a tuple of the components 
        (nD,nT,Dn2n,Tn2n)
    
    """
    mat_dict['D'].calc_dNdEs(I_E,rhoL_func)
    mat_dict['T'].calc_dNdEs(I_E,rhoL_func)
    nD   = frac_D*mat_dict['D'].elastic_dNdE
    nT   = frac_T*mat_dict['T'].elastic_dNdE
    Dn2n = frac_D*mat_dict['D'].n2n_dNdE
    Tn2n = frac_T*mat_dict['T'].n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

def DT_scatter_spec_w_ionkin(I_E : npt.NDArray, vbar : float, dv : float, rhoL_func : callable,
                             frac_D: float = frac_D_default, frac_T: float = frac_T_default
                             ) -> typing.Tuple[npt.NDArray,
                             typing.Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]]:
    """Calculates the single scattered neutron spectrum for DT given a 
    primary neutron spectrum of I_E from anisotropic areal density and including 
    ion velocities kinematics

    This requires the scattering matrices with ion kinematics to have been pre-calculated

    The primary neutron spectrum, I_E, is assumed to be on the same energy grid as 
    the incoming energy grid used to calculate the scattering matrices

    The areal density function rhoL_func needs to be a callable function with a 
    single argument (cosine[theta])

    Args:
        I_E (numpy.array): the neutron spectrum at Ein energies
        vbar (float) : mean velocity of the scattering ions
        dv (float) : standard deviation velocity of the scattering ions
        rhoL_func (callable): must be a single argument function f(x), 
        where x e [-1,1] and f(x) e [0,inf] and int f(x) dx = 1
        frac_D (float) : fraction of D in fuel
        frac_T (float) : fraction of T in fuel

    Returns:
        Tuple of numpy.arrays: the total scattered spectrum and a tuple of the components 
        (nD,nT,Dn2n,Tn2n)
    
    """
    rhoL_func = lambda x : np.ones_like(x)
    mat_dict['D'].calc_dNdEs(I_E,rhoL_func)
    mat_dict['T'].calc_dNdEs(I_E,rhoL_func)
    if(mat_dict['D'].vvec is None):
        dNdE_nD = mat_dict['D'].elastic_dNdE
    else:
        dNdE_nD = mat_dict['D'].matrix_interpolate_gaussian(mat_dict['D'].Eout,vbar,dv)
    if(mat_dict['T'].vvec is None):
        dNdE_nT = mat_dict['T'].elastic_dNdE
    else:
        dNdE_nT = mat_dict['T'].matrix_interpolate_gaussian(mat_dict['T'].Eout,vbar,dv)
    nD   = frac_D*dNdE_nD
    nT   = frac_T*dNdE_nT
    Dn2n = frac_D*mat_dict['D'].n2n_dNdE
    Tn2n = frac_T*mat_dict['T'].n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

def mat_scatter_spec(mat : typing.Type[sm.material_data],
                     I_E : npt.NDArray, rhoL_func : callable) -> npt.NDArray:
    """Calculates a material's single scattered neutron spectrum given a 
    primary neutron spectrum of I_E from anisotropic areal density

    This requires the scattering matrices to have been pre-calculated

    The primary neutron spectrum, I_E, is assumed to be on the same energy grid as 
    the incoming energy grid used to calculate the scattering matrices

    The areal density function rhoL_func needs to be a callable function with a 
    single argument (cosine[theta])

    Args:
        I_E (numpy.array): the neutron spectrum at Ein energies
        rhoL_func (callable): must be a single argument function f(x), 
        where x e [-1,1] and f(x) e [0,inf] and int f(x) dx = 1

    Returns:
        numpy.array: the total scattered spectrum
    
    """
    mat.calc_dNdEs(I_E,rhoL_func)
    total = mat.elastic_dNdE.copy()
    if(mat.l_n2n):
        total += mat.n2n_dNdE
    if(mat.l_inelastic):
        total += mat.inelastic_dNdE
    return total

###############################
# Full model fitting function #
###############################

def calc_DT_sigmabar(Ein : npt.NDArray, I_E : npt.NDArray,
                     frac_D: float = frac_D_default, frac_T: float = frac_T_default) -> float:
    """Calculates the spectral-averaged cross section for DT

    Args:
        Ein (numpy.array): the array on incoming neutron energies
        I_E (numpy.array): the neutron spectrum at Ein energies, assumed normalised
        frac_D (float) : fraction of D in fuel
        frac_T (float) : fraction of T in fuel

    Returns:
        float : the spectrally averaged total DT cross section
    """
    sigmabar = frac_D*np.trapz(mat_dict['D'].sigma_tot(Ein)*I_E,Ein)+frac_T*np.trapz(mat_dict['T'].sigma_tot(Ein)*I_E,Ein)
    return sigmabar

def rhoR_2_A1s(rhoR : typing.Union[float,npt.NDArray],
               frac_D: float = frac_D_default,frac_T: float = frac_T_default) -> typing.Union[float,npt.NDArray]:
    """Calculates the scattering amplitude given a DT areal density in kg/m^2

    Args:
        rhoR (float): the DT areal density in kg/m^2
        frac_D (float) : fraction of D in fuel
        frac_T (float) : fraction of T in fuel

    Returns:
        float : the scattering amplitude for single scattering
    """
    mbar  = (frac_D*sm.A_D+frac_T*sm.A_T)*Mn_kg
    A_1S = rhoR*(sigmabarn/mbar)
    return A_1S

def A1s_2_rhoR(A_1S : typing.Union[float,npt.NDArray],
               frac_D: float = frac_D_default,frac_T: float = frac_T_default) -> typing.Union[float,npt.NDArray]:
    """Calculates the DT areal density in kg/m^2 given a scattering amplitude 

    Args:
        A_1S (float): the scattering amplitude for single scattering
        frac_D (float) : fraction of D in fuel
        frac_T (float) : fraction of T in fuel

    Returns:
        float : the DT areal density in kg/m^2
    """
    mbar  = (frac_D*sm.A_D+frac_T*sm.A_T)*Mn_kg
    rhoR = A_1S/(sigmabarn/mbar)
    return rhoR
