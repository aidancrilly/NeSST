### ###################################### ###
### NeSST - Neutron Scattered Spectra Tool ###
### ###################################### ###

# Standard libraries
import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
# NeSST libraries
from NeSST.constants import *
import NeSST.collisions as col
import NeSST.spectral_model as sm

# Global variable defaults
col.classical_collisions = False
available_materials = list(sm.available_materials_dict.keys())
# Atomic fraction of D and T in scattering medium and source
frac_D_default = 0.5
frac_T_default = 0.5
# Set to True to calculate double scatters, set to False to not
# Note double scatter model assumes isotropic areal density for scattered neutrons - this is usually a poor approx.
DS_switch = False

# Energy units - MeV
# Temperature units - keV
# Velocity units - m/s

##########################################
# Primary spectral shapes & reactivities #
##########################################

# Gaussian "Brysk"
def Qb(Ein,mean,variance):
    spec = np.exp(-(Ein-mean)**2/2.0/variance)/np.sqrt(2*np.pi*variance)
    return spec

# Ballabio
def Qballabio(Ein,mean,variance):
    spec = np.exp(-2.0*mean*(np.sqrt(Ein)-np.sqrt(mean))**2/variance)/np.sqrt(2*np.pi*variance)
    return spec

# TT spectral shape
def dNdE_TT(E,Tion):
    return sm.TT_2dinterp(E,Tion)

def yield_from_dt_yield_ratio(reaction,dt_yield,Ti,frac_D=frac_D_default,frac_T=frac_T_default):
    ''' Reactivity ratio to predict yield from the DT yield assuming same volume and burn time
        rate_ij = (f_{i}*f_{j}*sigmav_{i,j}(T))/(1+delta_{i,j})  # dN/dVdt
        yield_ij = (rate_ij/rate_dt)*yield_dt

        Note that the TT reaction produces two neutrons.
    '''

    if sum([frac_D, frac_T]) != 1.:
        msg = (f'The frac_D ({frac_D_default}) and frac_T ({frac_T_default}) '
               'arguments on the yield_from_dt_yield_ration method do not sum to 1.')
        warnings.warn(msg)

    if reaction == 'tt':
        ratio = (0.5*frac_T*sm.reac_TT(Ti))/(frac_D*sm.reac_DT(Ti))
        ratio = 2.* ratio # Two neutrons are generated for each reaction
    if reaction == 'dd':
        ratio = (0.5*frac_D*sm.reac_DD(Ti))/(frac_T*sm.reac_DT(Ti))

    return ratio*dt_yield

###############################################################################
# Ballabio fits, see Table III of L. Ballabio et al 1998 Nucl. Fusion 38 1723 #
###############################################################################

# Returns the mean and variance based on Ballabio
# Tion in keV
def DTprimspecmoments(Tion):
    # Mean calculation
    a1 = 5.30509
    a2 = 2.4736e-3
    a3 = 1.84
    a4 = 1.3818

    mean_shift = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

    # keV to MeV
    mean_shift /= 1e3

    mean = 14.021 + mean_shift

    # Variance calculation
    omega0 = 177.259
    a1 = 5.1068e-4
    a2 = 7.6223e-3
    a3 = 1.78
    a4 = 8.7691e-5

    delta = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

    C = omega0*(1+delta)
    FWHM2    = C**2*Tion
    variance = FWHM2/(2.35482)**2
    # keV^2 to MeV^2
    variance /= 1e6

    return mean, variance

# Returns the mean and variance based on Ballabio
# Tion in keV
def DDprimspecmoments(Tion):
    # Mean calculation
    a1 = 4.69515
    a2 = -0.040729
    a3 = 0.47
    a4 = 0.81844

    mean_shift = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

    # keV to MeV
    mean_shift /= 1e3

    mean = 2.4495 + mean_shift

    # Variance calculation
    omega0 = 82.542
    a1 = 1.7013e-3
    a2 = 0.16888
    a3 = 0.49
    a4 = 7.9460e-4

    delta = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion

    C = omega0*(1+delta)
    FWHM2    = C**2*Tion
    variance = FWHM2/(2.35482)**2
    # keV^2 to MeV^2
    variance /= 1e6

    return mean, variance

#######################################
# DT scattered spectra initialisation #
#######################################

def init_DT_scatter(Eout,Ein):
    sm.mat_D.init_energy_grids(Eout,Ein)
    sm.mat_T.init_energy_grids(Eout,Ein)
    sm.mat_D.init_station_scatter_matrices()
    sm.mat_T.init_station_scatter_matrices()

def init_DT_ionkin_scatter(varr,nT=False,nD=False):
    if(nT):
        if(sm.mat_T.Ein is None):
            print("nT - Needed to initialise energy grids - see init_DT_scatter")
        else:
            sm.mat_T.full_scattering_matrix_create(varr)
    if(nD):
        if(sm.mat_D.Ein is None):
            print("nD - Needed to initialise energy grids - see init_DT_scatter")
        else:
            sm.mat_D.full_scattering_matrix_create(varr)

def calc_DT_ionkin_primspec_rhoL_integral(I_E,rhoL_func=None,nT=False,nD=False):
    if(nT):
        if(sm.mat_T.vvec is None):
            print("nT - Needed to initialise velocity grid - see init_DT_ionkin_scatter")
        else:
            if(rhoL_func is not None):
                sm.mat_T.scattering_matrix_apply_rhoLfunc(rhoL_func)
            sm.mat_T.matrix_primspec_int(I_E)
    if(nD):
        if(sm.mat_D.vvec is None):
            print("nD - Needed to initialise velocity grid - see init_DT_ionkin_scatter")
        else:
            if(rhoL_func is not None):
                sm.mat_D.scattering_matrix_apply_rhoLfunc(rhoL_func)
            sm.mat_D.matrix_primspec_int(I_E)


###################################
# General material initialisation #
###################################
def init_mat_scatter(Eout,Ein,mat_label):
    mat = sm.available_materials_dict[mat_label]
    mat.init_energy_grids(Eout,Ein)
    mat.init_station_scatter_matrices()
    return mat

###########################################
#   Single Evalutation Scattered Spectra  #
###########################################

def DT_sym_scatter_spec(I_E,frac_D=frac_D_default,frac_T=frac_T_default):
    rhoL_func = lambda x : np.ones_like(x)
    sm.mat_D.calc_dNdEs(I_E,rhoL_func)
    sm.mat_T.calc_dNdEs(I_E,rhoL_func)
    nD   = frac_D*sm.mat_D.elastic_dNdE
    nT   = frac_T*sm.mat_T.elastic_dNdE
    Dn2n = frac_D*sm.mat_D.n2n_dNdE
    Tn2n = frac_T*sm.mat_T.n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

def DT_scatter_spec_w_ionkin(I_E,vbar,dv,rhoL_func,frac_D=frac_D_default,frac_T=frac_T_default):
    rhoL_func = lambda x : np.ones_like(x)
    sm.mat_D.calc_dNdEs(I_E,rhoL_func)
    sm.mat_T.calc_dNdEs(I_E,rhoL_func)
    if(sm.mat_D.vvec is None):
        dNdE_nD = sm.mat_D.elastic_dNdE
    else:
        dNdE_nD = sm.mat_D.matrix_interpolate_gaussian(sm.mat_D.Eout,vbar,dv)
    if(sm.mat_T.vvec is None):
        dNdE_nT = sm.mat_T.elastic_dNdE
    else:
        dNdE_nT = sm.mat_T.matrix_interpolate_gaussian(sm.mat_T.Eout,vbar,dv)
    nD   = frac_D*dNdE_nD
    nT   = frac_T*dNdE_nT
    Dn2n = frac_D*sm.mat_D.n2n_dNdE
    Tn2n = frac_T*sm.mat_T.n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

def DT_asym_scatter_spec(I_E,rhoL_func,frac_D=frac_D_default,frac_T=frac_T_default):
    sm.mat_D.calc_dNdEs(I_E,rhoL_func)
    sm.mat_T.calc_dNdEs(I_E,rhoL_func)
    nD   = frac_D*sm.mat_D.elastic_dNdE
    nT   = frac_T*sm.mat_T.elastic_dNdE
    Dn2n = frac_D*sm.mat_D.n2n_dNdE
    Tn2n = frac_T*sm.mat_T.n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

def mat_scatter_spec(mat,I_E,rhoL_func):
    mat.calc_dNdEs(I_E,rhoL_func)
    total = mat.elastic_dNdE
    if(mat.l_n2n):
        total += mat.n2n_dNdE
    if(mat.inelastic):
        total += mat.inelastic_dNdE
    return total

###############################
# Full model fitting function #
###############################

def calc_DT_sigmabar(Ein,I_E,frac_D=frac_D_default,frac_T=frac_T_default):
    sigmabar = frac_D*np.trapz(sm.mat_D.sigma_tot(Ein)*I_E,Ein)+frac_T*np.trapz(sm.mat_T.sigma_tot(Ein)*I_E,Ein)
    return sigmabar

def rhoR_2_A1s(rhoR,frac_D=frac_D_default,frac_T=frac_T_default):
    # Mass of neutron in milligrams
    mn_mg = 1.674927485e-21
    mbar  = (frac_D*sm.A_D+frac_T*sm.A_T)*mn_mg
    # barns to cm^2
    sigmabarn = 1e-24
    A_1S = rhoR*(sigmabarn/mbar)
    return A_1S

def A1s_2_rhoR(A_1S,frac_D=frac_D_default,frac_T=frac_T_default):
    # Mass of neutron in milligrams
    mn_mg = 1.674927485e-21
    mbar  = (frac_D*sm.A_D+frac_T*sm.A_T)*mn_mg
    # barns to cm^2
    sigmabarn = 1e-24
    rhoR = A_1S/(sigmabarn/mbar)
    return rhoR