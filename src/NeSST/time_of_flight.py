from NeSST.constants import *
from NeSST.collisions import *
import numpy as np
from scipy.interpolate import interp1d

def get_unity_sensitivity():
    def unity_sensitivity(En):
        return np.ones_like(En)
    return unity_sensitivity

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

class nToF:

    def __init__(self,distance,sensitivity,instrument_response_function
                 ,normtime_start=5.0,normtime_end=22.0,normtime_N=1000,detector_normtime=None):
        self.distance = distance
        self.sensitivity = sensitivity
        self.instrument_response_function = instrument_response_function

        if(detector_normtime is None):
            self.detector_normtime = np.linspace(normtime_start,normtime_end,normtime_N)
        else:
            self.detector_normtime = detector_normtime
        self.detector_time     = self.detector_normtime*self.distance/c
        # Init instrument response values to None
        self.dEdt = None
        self.sens = None
        self.R    = None
        self.En   = None

    def compute_instrument_reponse(self,En):
        self.En_dNdE = En
        self.dEdt = Jacobian_dEdnorm_t(En,Mn)

        self.En_det = beta_2_Ekin(1.0/self.detector_normtime,Mn)
        
        self.sens = self.sensitivity(self.En_det)
        self.R = self.instrument_response_function(self.detector_time,self.En_det)

    def get_dNdt(self,dNdE):
        dNdt = dNdE*self.dEdt
        dNdt_interp = interp1d(self.En_dNdE,dNdt,bounds_error=False,fill_value=0.0)
        return dNdt_interp(self.En_det)

    def get_signal(self,En,dNdE):
        if(not np.array_equal(En,self.En)):
            self.compute_instrument_reponse(En)
        dNdt = self.get_dNdt(dNdE)

        return self.detector_time,self.detector_normtime,np.matmul(self.R,self.sens*dNdt)
