### ###################################### ###
### NeSST - Neutron Scattered Spectra Tool ###
### ###################################### ###

# Standard libraries
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
# Elastic collisions scattering kernel &
# differential cross sections
import collisions as col
import spectral_model as sm
from Constants import *

# Energy units - MeV
# Temperature units - keV
# Velocity units - m/s

# Global variable defaults
col.classical_collisions = False
# Atomic fraction of D and T in scattering medium and source
frac_D_default = 0.5
frac_T_default = 0.5
# Set to True to calculate double scatters, set to False to not
# Note double scatter model assumes isotropic areal density for scattered neutrons - this is usually a poor approx.
DS_switch = False

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

###########################################
#   Single Evalutation Scattered Spectra  #
###########################################

def sym_scatter_spec(I_E,frac_D=frac_D_default,frac_T=frac_T_default):
    rhoL_func = lambda x : np.ones_like(x)
    sm.mat_D.calc_station_elastic_dNdE(I_E,rhoL_func)
    sm.mat_T.calc_station_elastic_dNdE(I_E,rhoL_func)
    sm.mat_D.calc_n2n_dNdE(I_E,rhoL_func)
    sm.mat_T.calc_n2n_dNdE(I_E,rhoL_func)
    nD   = frac_D*sm.mat_D.elastic_dNdE
    nT   = frac_T*sm.mat_T.elastic_dNdE
    Dn2n = frac_D*sm.mat_D.n2n_dNdE
    Tn2n = frac_T*sm.mat_T.n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

def asym_scatter_spec(I_E,rhoL_func,frac_D=frac_D_default,frac_T=frac_T_default):
    sm.mat_D.calc_station_elastic_dNdE(I_E,rhoL_func)
    sm.mat_T.calc_station_elastic_dNdE(I_E,rhoL_func)
    sm.mat_D.calc_n2n_dNdE(I_E,rhoL_func)
    sm.mat_T.calc_n2n_dNdE(I_E,rhoL_func)
    nD   = frac_D*sm.mat_D.elastic_dNdE
    nT   = frac_T*sm.mat_T.elastic_dNdE
    Dn2n = frac_D*sm.mat_D.n2n_dNdE
    Tn2n = frac_T*sm.mat_T.n2n_dNdE
    total = nD+nT+Dn2n+Tn2n
    return total,(nD,nT,Dn2n,Tn2n)

##################################
# Splining background components #
##################################
# These will be tidied 

def calc_all_splines(E_full_arr,Ein,I_E,P1,frac_D=frac_D_default,frac_T=frac_T_default):
    # DT scattering 
    sm.init_n2n_ddxs_mode1(E_full_arr,Ein,I_E,P1)
    Tn2nspline   = interp2d(E_full_arr,P1,frac_T*sm.Tn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    Dn2nspline   = interp2d(E_full_arr,P1,frac_D*sm.Dn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    nTspline     = interp2d(E_full_arr,P1,frac_T*sm.nTspec(E_full_arr,Ein,I_E,A_T,P1).T,bounds_error=False,fill_value=0.0)
    nDspline     = interp2d(E_full_arr,P1,frac_D*sm.nDspec(E_full_arr,Ein,I_E,A_D,P1).T,bounds_error=False,fill_value=0.0)
    return nTspline,nDspline,Dn2nspline,Tn2nspline

def calc_splines_w_TT(E_full_arr,Ein,I_E,P1,Tion,frac_D=frac_D_default,frac_T=frac_T_default):
    # TT primaries (note this is area normalized)
    I_TT         = sm.TT_2dinterp(E_full_arr,Tion)
    TT_spline    = interp1d(E_full_arr,I_TT)
    TT_2_DT_reac = yield_from_dt_yield_ratio('tt',1.0,Tion)
    # DT scattering 
    sm.init_n2n_ddxs_mode1(E_full_arr,Ein,I_E,P1)
    n2nspline    = interp2d(E_full_arr,P1,frac_T*sm.Tn2n_ddx.rgrid_P1.T+frac_D*sm.Dn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    nDspline     = interp2d(E_full_arr,P1,frac_D*sm.nDspec(E_full_arr,Ein,I_E,A_D,P1).T,bounds_error=False,fill_value=0.0)
    # TT scattered spline
    I_TT_1S_nT   = frac_T*sm.nTspec(E_full_arr,E_full_arr,I_TT,A_T)
    I_TT_1S_nD   = frac_D*sm.nDspec(E_full_arr,E_full_arr,I_TT,A_D)
    I_TT_1S      = TT_2_DT_reac*(I_TT_1S_nT+I_TT_1S_nD)
    TT_1Sspline  = interp1d(E_full_arr,I_TT_1S,bounds_error=False,fill_value=0.0)
    return (nDspline,n2nspline,TT_spline,TT_1Sspline)

def calc_all_splines_w_TT(E_full_arr,Ein,I_E,P1,Tion,frac_D=frac_D_default,frac_T=frac_T_default):
    # TT primaries (note this is area normalized)
    I_TT         = sm.TT_2dinterp(E_full_arr,Tion)
    TT_spline    = interp1d(E_full_arr,I_TT)
    TT_2_DT_reac = yield_from_dt_yield_ratio('tt',1.0,Tion)
    # DT scattering 
    sm.init_n2n_ddxs_mode1(E_full_arr,Ein,I_E,P1)
    Tn2nspline   = interp2d(E_full_arr,P1,frac_T*sm.Tn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    Dn2nspline   = interp2d(E_full_arr,P1,frac_D*sm.Dn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    nTspline     = interp2d(E_full_arr,P1,frac_T*sm.nTspec(E_full_arr,Ein,I_E,A_T,P1).T,bounds_error=False,fill_value=0.0)
    nDspline     = interp2d(E_full_arr,P1,frac_D*sm.nDspec(E_full_arr,Ein,I_E,A_D,P1).T,bounds_error=False,fill_value=0.0)
    # TT scattered spline
    I_TT_1S_nT   = frac_T*sm.nTspec(E_full_arr,E_full_arr,I_TT,A_T)
    I_TT_1S_nD   = frac_D*sm.nDspec(E_full_arr,E_full_arr,I_TT,A_D)
    I_TT_1S      = TT_2_DT_reac*(I_TT_1S_nT+I_TT_1S_nD)
    TT_1Sspline  = interp1d(E_full_arr,I_TT_1S,bounds_error=False,fill_value=0.0)
    return (nTspline,nDspline,Dn2nspline,Tn2nspline,TT_spline,TT_1Sspline)

def calc_all_splines_w_DD(E_full_arr,Ein,I_DT,I_DD,P1,Tion,frac_D=frac_D_default,frac_T=frac_T_default):
    # TT and DD primaries
    I_TT         = sm.TT_2dinterp(E_full_arr,Tion)
    TT_2_DT_reac = yield_from_dt_yield_ratio('tt',1.0,Tion)
    DD_2_DT_reac = yield_from_dt_yield_ratio('dd',1.0,Tion)
    prim_spline  = interp1d(E_full_arr,TT_2_DT_reac*I_TT+DD_2_DT_reac*I_DD)
    # DT scattering 
    sm.init_n2n_ddxs_mode1(E_full_arr,Ein,I_DT,P1)
    Dn2nspline   = interp2d(E_full_arr,P1,frac_D*sm.Dn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    Tn2nspline   = interp2d(E_full_arr,P1,frac_T*sm.Tn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    nTspline     = interp2d(E_full_arr,P1,frac_T*sm.nTspec(E_full_arr,Ein,I_DT,A_T,P1).T,bounds_error=False,fill_value=0.0)
    nDspline     = interp2d(E_full_arr,P1,frac_D*sm.nDspec(E_full_arr,Ein,I_DT,A_D,P1).T,bounds_error=False,fill_value=0.0)
    # TT and DD scattered spline
    I_1S_nT   = interp2d(E_full_arr,P1,frac_T*sm.nTspec(E_full_arr,E_full_arr,TT_2_DT_reac*I_TT+DD_2_DT_reac*I_DD,A_T,P1).T,bounds_error=False,fill_value=0.0)
    I_1S_nD   = interp2d(E_full_arr,P1,frac_D*sm.nDspec(E_full_arr,E_full_arr,TT_2_DT_reac*I_TT+DD_2_DT_reac*I_DD,A_D,P1).T,bounds_error=False,fill_value=0.0)
    return (nTspline,nDspline,Tn2nspline,Dn2nspline,prim_spline,I_1S_nT,I_1S_nD)

def calc_splines_w_DD(E_full_arr,Ein,I_DT,I_DD,P1,Tion,frac_D=frac_D_default,frac_T=frac_T_default):
    # TT and DD primaries
    I_TT         = sm.TT_2dinterp(E_full_arr,Tion)
    TT_2_DT_reac = yield_from_dt_yield_ratio('tt',1.0,Tion)
    DD_2_DT_reac = yield_from_dt_yield_ratio('dd',1.0,Tion)
    prim_spline  = interp1d(E_full_arr,TT_2_DT_reac*I_TT+DD_2_DT_reac*I_DD)
    # DT scattering 
    sm.init_n2n_ddxs_mode1(E_full_arr,Ein,I_DT,P1)
    n2nspline    = interp2d(E_full_arr,P1,frac_T*sm.Tn2n_ddx.rgrid_P1.T+frac_D*sm.Dn2n_ddx.rgrid_P1.T,bounds_error=False,fill_value=0.0)
    nDspline     = interp2d(E_full_arr,P1,frac_D*sm.nDspec(E_full_arr,Ein,I_DT,A_D,P1).T,bounds_error=False,fill_value=0.0)
    # TT and DD scattered spline
    I_1S_nT   = frac_T*sm.nTspec(E_full_arr,E_full_arr,TT_2_DT_reac*I_TT+DD_2_DT_reac*I_DD,A_T)
    I_1S_nD   = frac_D*sm.nDspec(E_full_arr,E_full_arr,TT_2_DT_reac*I_TT+DD_2_DT_reac*I_DD,A_D)
    I_1S      = (I_1S_nT+I_1S_nD)
    I_1Sspline  = interp1d(E_full_arr,I_1S,bounds_error=False,fill_value=0.0)
    return (nDspline,n2nspline,prim_spline,I_1Sspline)

#####################################################
# Inclusion of ion velocities to scattering kernels #
#####################################################

# Inline calculation

# Integrand of Eq. 8 in A. J. Crilly 2019 PoP
def integrand(Ein,Eout,muin,vf,Ein_vec,I_E,reac_type):
    # Reverse velocity direction so +ve vf is implosion
    # Choose this way round so vf is +ve if shell coming TOWARDS detector
    vf    = -vf
    if  (reac_type == "nT"):
        A = sm.A_T
    elif(reac_type == "nD"):
        A = sm.A_D
    else:
        # print('WARNING: Unkown reaction type %s, setting edge to zeros'%reac_type)
        return np.zeros(np.shape(col.mu_out(A_T,Ein,Eout,vf)))
    muout = col.mu_out(A,Ein,Eout,vf)
    jacob = col.g(A,Ein,Eout,muin,muout,vf)
    flux_change = col.flux_change(Ein,muin,vf)
    dsdO = sm.dsigdOmega(A,Ein,Eout,Ein_vec,muin,muout,vf,reac_type)
    return flux_change*dsdO*jacob*I_E # Integrand of Eq. 8 in A. J. Crilly 2019 PoP

# Generates a matrix of the edge spectrum generated off a scatter of ion with velocity
# v. Note this takes care of the integration with the primary source spectrum I_E
# See equation 8 of Crilly 2019 PoP, E' integration performed within this subroutine
def matrix_calc(E,v,Ein,I_E,reac_type):
    # Generates a matrix of the edge spectrum generated off a scatter of ion with velocity
    # v. Note this takes care of the integration with the primary source spectrum I_E
    vv,EEo,EEi = np.meshgrid(v,E,Ein)
    integral   = integrand(EEi,EEo,1.0,vv,Ein,I_E,reac_type)
    # Integrate the source spectrum component
    M          = np.trapz(integral,Ein,axis=-1)
    return M

# Perform 1D linear interpolation in energy space and integrate over Gaussian in velocity space
def interpolate_matrix(M,E,vbar,dv,v_arr,E_arr):
    gauss       = np.exp(-(v_arr-vbar)**2/2.0/(dv**2))/np.sqrt(2*np.pi)/dv
    idx         = np.argmax(E_arr > E)
    frac        = (E_arr[idx]-E)/(E_arr[idx]-E_arr[idx-1])
    interp_in_E = (1-frac)*M[idx,:]+frac*M[idx-1,:]
    return np.trapz(gauss*interp_in_E,v_arr) # this is the integral of veloctiies

# Perform 1D linear interpolation in energy space and integrate over provided PDF in velocity space
def interpolate_matrix_w_PDF(M,E,PDF,v_arr,E_arr):
    idx         = np.argmax(E_arr > E)
    frac        = (E_arr[idx]-E)/(E_arr[idx]-E_arr[idx-1])
    interp_in_E = (1-frac)*M[idx,:]+frac*M[idx-1,:]
    return np.trapz(PDF*interp_in_E,v_arr) # this is the integral of veloctiies

# Perform 2D linear interpolation in energy and velocity space
def interpolate_2D_matrix(M,E,vbar,v_arr,E_arr):
    idx1  = np.argmax(E_arr > E)
    frac1 = (E_arr[idx1]-E)/(E_arr[idx1]-E_arr[idx1-1])
    idx2  = np.argmax(v_arr > vbar)
    frac2 = (v_arr[idx2]-vbar)/(v_arr[idx2]-v_arr[idx2-1])
    interp_in_E1 = (1-frac1)*M[idx1,idx2]+frac1*M[idx1-1,idx2]
    interp_in_E2 = (1-frac1)*M[idx1,idx2-1]+frac1*M[idx1-1,idx2-1]
    interp_in_2D = (1-frac2)*interp_in_E1+frac2*interp_in_E2
    return interp_in_2D

# Calculate the scattered spectrum over outgoing energy array E
# Matrix in energy and velocity is interpolated in energy and integrated in velocity
# via the appropriate scattering ion velocity distribution
# Can hopefully replace for loop at some point
def generate_velocity_weighted_spectrum(M,E,vbar,dv,v_arr,E_arr):
    I_bs = np.zeros(len(E))
    for i,e in enumerate(E):
        I_bs[i]  = interpolate_matrix(M,e,vbar,dv,v_arr,E_arr)
    return I_bs

def generate_velocity_weighted_spectrum_w_PDF(M,E,PDF,v_arr,E_arr):
    I_bs = np.zeros(len(E))
    for i,e in enumerate(E):
        I_bs[i]  = interpolate_matrix_w_PDF(M,e,PDF,v_arr,E_arr)
    return I_bs

def generate_single_velocity_spectrum(M,E,v_bar,v_arr,E_arr):
    I_bs = np.zeros(len(E))
    for i,e in enumerate(E):
        I_bs[i]  = interpolate_2D_matrix(M,e,v_bar,v_arr,E_arr)
    return I_bs

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