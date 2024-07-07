from NeSST.constants import *
from NeSST.collisions import *
from NeSST.utils import *
from NeSST.endf_interface import retrieve_total_cross_section_from_ENDF_file
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import erf

from dataclasses import dataclass,field
from typing import List

@dataclass
class LOS_material_component:
    number_fraction : float
    A : float
    ENDF_file : str

@dataclass
class LOS_material:
    density : float
    ntot    : float = field(init=False)
    mavg    : float = field(init=False)
    length  : float
    components : List[LOS_material_component]

    def __post_init__(self):
        self.mavg = 0.0
        for comp in self.components:
            self.mavg += comp.number_fraction*comp.A
        self.mavg *= sc.atomic_mass
        self.ntot = self.density/self.mavg

def get_LOS_attenuation(LOS_materials : List[LOS_material]):
    tau_interp_list = []
    for LOS_material in LOS_materials:
        ntot_barn   = 1e-28*LOS_material.ntot
        L           = LOS_material.length
        for LOS_component in LOS_material.components:
            ncomp = ntot_barn*LOS_component.number_fraction
            E,sigma_tot = retrieve_total_cross_section_from_ENDF_file(LOS_component.ENDF_file)
            tau_interp  = interpolate_1d(E,L*ncomp*sigma_tot,method='linear')
            tau_interp_list.append(tau_interp)

    def LOS_attenuation(E):
        total_tau = tau_interp_list[0](E)
        for i in range(1,len(tau_interp_list)):
            total_tau += tau_interp_list[i](E)
        transmission = np.exp(-total_tau)
        return transmission
    
    return LOS_attenuation

def get_power_law_NLO(p,Enorm=E0_DT):
    A = mat_dict['H'].sigma(Enorm)*Enorm**p
    def power_law_NLO(E):
        return mat_dict['H'].sigma(E)*E**p/A
    return power_law_NLO

def get_Verbinski_NLO(Enorm=E0_DT):
    V_E,V_L = np.loadtxt(data_dir+'VerbinskiLproton.csv',delimiter=',',unpack=True)
    cumulative_L = cumtrapz(y=np.insert(V_L,0,0.0),x=np.insert(V_E,0,0.0))
    L_integral = interpolate_1d(np.insert(V_E,0,0.0)*1e6,np.insert(cumulative_L,0,0.0),method='cubic')

    def Verbinski_NLO(E):
        return mat_dict['H'].sigma(E)*L_integral(E)/E
    
    A = Verbinski_NLO(Enorm)

    def norm_Verbinski_NLO(E):
        return Verbinski_NLO(E)/A
    
    return norm_Verbinski_NLO

def get_unity_sensitivity():
    def unity_sensitivity(En):
        return np.ones_like(En)
    return unity_sensitivity

def combine_detector_sensitivities(model_list):
    def total_sensitivity(E):
        sensitivity = model_list[0](E)
        for i in range(1,len(model_list)):
            sensitivity *= model_list[i](E)
        return sensitivity
    return total_sensitivity

def get_transit_time_tophat_IRF(scintillator_thickness):
    def transit_time_tophat_IRF(t_detected,En):
        vn = Ekin_2_beta(En,Mn)*c
        t_transit = scintillator_thickness/vn
        tt_d,tt_a = np.meshgrid(t_detected,t_detected,indexing='ij')
        _,tt_t    = np.meshgrid(t_detected,t_transit,indexing='ij')
        R = np.eye(t_detected.shape[0])+np.heaviside(tt_d-tt_a,0.0)-np.heaviside(tt_d-(tt_a+tt_t),1.0)
        R_norm = np.sum(R,axis=1)
        # Catch the zeros
        R_norm[R_norm == 0] = 1
        R /= R_norm[:,None]
        return R
    return transit_time_tophat_IRF

def get_transit_time_tophat_w_decayingGaussian_IRF(scintillator_thickness,gaussian_FWHM,decay_time):
    """
    
    See: https://pubs.aip.org/aip/rsi/article/68/1/610/1070804/Interpretation-of-neutron-time-of-flight-signals

    """
    sig = gaussian_FWHM/2.355
    shift_t = 2*sig
    def filter(t):
        t_shift = t-shift_t
        erf_arg = (t_shift-sig**2/decay_time)/np.sqrt(2*sig**2)
        gauss = np.exp(-t_shift/decay_time)*np.exp(0.5*sig**2/decay_time**2)/(2*decay_time)*(1+erf(erf_arg))
        gauss[t < 0] = 0.0
        return gauss
    _tophat_IRF = get_transit_time_tophat_IRF(scintillator_thickness)
    def transit_time_tophat_w_decayingGaussian_IRF(t_detected,En):
        R = _tophat_IRF(t_detected,En)
        t_filter = t_detected-0.5*(t_detected[-1]+t_detected[0])
        filt = filter(t_filter)
        R = np.apply_along_axis(lambda m : np.convolve(m,filt,mode='same'), axis=0, arr=R)
        R_norm = np.sum(R,axis=1)
        # Catch the zeros
        R_norm[R_norm == 0] = 1
        R /= R_norm[:,None]
        return R
    return transit_time_tophat_w_decayingGaussian_IRF

def get_transit_time_tophat_w_tGaussian_IRF(scintillator_thickness,gaussian_FWHM,tgaussian_peak_pos):
    sig = gaussian_FWHM/2.355
    mu  = (tgaussian_peak_pos**2-sig**2)/tgaussian_peak_pos
    def filter(t):
        gauss = t*np.exp(-0.5*((t-mu)/sig)**2)
        gauss[t < 0] = 0.0
        return gauss
    _tophat_IRF = get_transit_time_tophat_IRF(scintillator_thickness)
    def transit_time_tophat_w_tGaussian_IRF(t_detected,En):
        R = _tophat_IRF(t_detected,En)
        t_filter = t_detected-0.5*(t_detected[-1]+t_detected[0])
        filt = filter(t_filter)
        R = np.apply_along_axis(lambda m : np.convolve(m,filt,mode='same'), axis=0, arr=R)
        R_norm = np.sum(R,axis=1)
        # Catch the zeros
        R_norm[R_norm == 0] = 1
        R /= R_norm[:,None]
        return R
    return transit_time_tophat_w_tGaussian_IRF

class nToF:

    def __init__(self,distance,sensitivity,instrument_response_function
                 ,normtime_start=5.0,normtime_end=20.0,normtime_N=2048,detector_normtime=None):
        self.distance = distance
        self.sensitivity = sensitivity
        self.instrument_response_function = instrument_response_function

        if(detector_normtime is None):
            self.detector_normtime = np.linspace(normtime_start,normtime_end,normtime_N)
        else:
            self.detector_normtime = detector_normtime
        self.detector_time     = self.detector_normtime*self.distance/c
        # Init instrument response values to None
        self.dEdt    = None
        self.sens    = None
        self.R       = None
        self.En_dNdE = None

    def compute_instrument_reponse(self,En):
        self.En_dNdE = En
        self.dEdt = Jacobian_dEdnorm_t(En,Mn)

        self.En_det = beta_2_Ekin(1.0/self.detector_normtime,Mn)
        
        self.sens = self.sensitivity(self.En_det)
        self.R = self.instrument_response_function(self.detector_time,self.En_det)

    def get_dNdt(self,dNdE):
        dNdt = dNdE*self.dEdt
        dNdt_interp = interpolate_1d(self.En_dNdE,dNdt,bounds_error=False,fill_value=0.0)
        return dNdt_interp(self.En_det)

    def get_signal(self,En,dNdE):
        if(not np.array_equal(En,self.En_dNdE)):
            self.compute_instrument_reponse(En)
        dNdt = self.get_dNdt(dNdE)

        return self.detector_time,self.detector_normtime,np.matmul(self.R,self.sens*dNdt)
    
    def get_signal_no_IRF(self,En,dNdE):
        if(not np.array_equal(En,self.En_dNdE)):
            self.En_dNdE = En
            self.dEdt = Jacobian_dEdnorm_t(En,Mn)
        dNdt = self.get_dNdt(dNdE)

        return self.detector_time,self.detector_normtime,dNdt
