# Backend of spectral model

import numpy as np

from NeSST.constants import *
from NeSST.utils import *
from NeSST.endf_interface import retrieve_ENDF_data
import NeSST.collisions as col
import NeSST.cross_sections as xs

##################
# Material class #
##################

# A values needed for scattering kinematics
A_H  = Mp/Mn
A_D  = Md/Mn
A_T  = Mt/Mn
A_C  = MC/Mn
A_Be = MBe/Mn

def unity(x):
    return np.ones_like(x)

class material_data:

    def __init__(self,label,json):
        self.label = label

        self.json = json
        ENDF_data = retrieve_ENDF_data(self.json)

        self.A = ENDF_data['A']

        if(ENDF_data['interactions'].total):
            self.sigma_tot = interpolate_1d(ENDF_data['total_xsec']['E'],ENDF_data['total_xsec']['sig'],method='linear',bounds_error=False,fill_value=0.0)

        if(ENDF_data['interactions'].elastic):
            self.sigma = interpolate_1d(ENDF_data['elastic_xsec']['E'],ENDF_data['elastic_xsec']['sig'],method='linear',bounds_error=False,fill_value=0.0)

            self.elastic_legendre = ENDF_data['elastic_dxsec']['legendre']
            if(self.elastic_legendre):
                self.legendre_dx_spline = [unity]
                for i in range(ENDF_data['elastic_dxsec']['N_l']):
                    self.legendre_dx_spline.append(interpolate_1d(ENDF_data['elastic_dxsec']['E'],ENDF_data['elastic_dxsec']['a_l'][:,i],method='linear',bounds_error=False,fill_value=0.0))
            else:
                self.elastic_SDX_table = ENDF_data['elastic_dxsec']['SDX']

        self.l_n2n = ENDF_data['interactions'].n2n
        if(ENDF_data['interactions'].n2n):
            if(ENDF_data['n2n_dxsec']['LAW'] == 6):
                self.n2n_ddx = xs.doubledifferentialcrosssection_LAW6(ENDF_data['n2n_xsec'],ENDF_data['n2n_dxsec'])
            elif(ENDF_data['n2n_dxsec']['LAW'] == 7):
                self.n2n_ddx = xs.doubledifferentialcrosssection_data(ENDF_data['n2n_xsec'],ENDF_data['n2n_dxsec'])

        self.l_inelastic = ENDF_data['interactions'].inelastic
        if(ENDF_data['interactions'].inelastic):
            self.n_inelastic = ENDF_data['n_inelastic']

            self.isigma = []
            self.inelasticQ = []
            self.inelastic_legendre = []
            self.legendre_idx_spline = []
            self.inelastic_SDX_table = []

            for i_inelastic in range(self.n_inelastic):
                xsec_table = ENDF_data[f'inelastic_xsec_n{i_inelastic+1}']
                self.isigma.append(interpolate_1d(xsec_table['E'],xsec_table['sig'],method='linear',bounds_error=False,fill_value=0.0))

                dxsec_table = ENDF_data[f'inelastic_dxsec_n{i_inelastic+1}']
                self.inelasticQ.append(dxsec_table['Q'])

                self.inelastic_legendre.append(dxsec_table['legendre'])
                if(dxsec_table['legendre']):
                    idx_spline = [unity]
                    for i in range(dxsec_table['N_l']):
                        idx_spline.append(interpolate_1d(dxsec_table['E'],dxsec_table['a_l'][:,i],method='linear',bounds_error=False,fill_value=0.0))
                    self.legendre_idx_spline.append(idx_spline)
                    self.inelastic_SDX_table.append(None)
                else:
                    self.legendre_idx_spline.append(None)
                    self.inelastic_SDX_table.append(dxsec_table['SDX'])

        self.Ein  = None       
        self.Eout = None
        self.vvec = None

    ############################################
    # Stationary ion scattered spectral shapes #
    ############################################

    def init_energy_grids(self,Eout,Ein):
        self.Eout = Eout
        self.Ein  = Ein

    def init_station_scatter_matrices(self,Nm=100):
        self.init_station_elastic_scatter()
        if(self.l_n2n):
            self.init_n2n_ddxs(Nm)
        if(self.l_inelastic):
            self.init_station_inelastic_scatter()

    # Elastic scatter matrix
    def init_station_elastic_scatter(self):
        Ei,Eo  = np.meshgrid(self.Ein,self.Eout)
        muc    = col.muc(self.A,Ei,Eo,1.0,-1.0,0.0)
        sigma  = self.sigma(self.Ein)
        self.elastic_mu0 = col.mu_out(self.A,Ei,Eo,0.0)
        if(self.elastic_legendre):
            Tlcoeff,Nl     = xs.interp_Tlcoeff(self.legendre_dx_spline,self.Ein)
            Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
            dsdO = xs.diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
        else:
            dsdO = xs.diffxsec_table_eval(sigma,muc,Ei,self.elastic_SDX_table)
        jacob = col.g(self.A,Ei,Eo,1.0,-1.0,0.0)
        self.elastic_dNdEdmu = jacob*dsdO

    # Inelastic scatter matrix
    # Currently uses classical kinematics
    def init_station_inelastic_scatter(self):
        Ei,Eo  = np.meshgrid(self.Ein,self.Eout)
        self.inelastic_mu0 = []
        self.inelastic_dNdEdmu = []
        for i_inelastic in range(self.n_inelastic):
            kin_a2 = (self.A/(self.A+1))**2*(1.0+(self.A+1)/self.A*self.inelasticQ[i_inelastic]/Ei)
            kin_a2_safe = kin_a2.copy()
            kin_a2_safe[kin_a2_safe < 0.0] = 1.0
            kin_a  = np.sqrt(kin_a2_safe)
            kin_b  = 1.0/(self.A+1)
            muc    = ((Eo/Ei)-kin_a**2-kin_b**2)/(2*kin_a*kin_b)
            sigma  = self.isigma[i_inelastic](self.Ein)
            inelastic_mu0 = (np.sqrt(Eo/Ei)-(kin_a**2-kin_b**2)*np.sqrt(Ei/Eo))/(2*kin_b)
            inelastic_mu0[kin_a2 < 0.0] = 0.0
            self.inelastic_mu0.append(inelastic_mu0)

            if(self.inelastic_legendre[i_inelastic]):
                Tlcoeff,Nl     = xs.interp_Tlcoeff(self.legendre_idx_spline[i_inelastic],self.Ein)
                Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
                dsdO = xs.diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
            else:
                dsdO = xs.diffxsec_table_eval(sigma,muc,Ei,self.inelastic_SDX_table[i_inelastic])

            jacob = 2.0/((kin_a+kin_b)**2-(kin_a-kin_b)**2)/Ei
            inelastic_dNdEdmu = jacob*dsdO
            
            inelastic_dNdEdmu[kin_a2 < 0.0] = 0.0

            self.inelastic_dNdEdmu.append(inelastic_dNdEdmu)

    def init_n2n_ddxs(self,Nm=100):
        self.n2n_mu = np.linspace(-1.0,1.0,Nm)
        self.n2n_ddx.regular_grid(self.Ein,self.n2n_mu,self.Eout)

    def calc_dNdEs(self,I_E,rhoL_func):
        self.calc_station_elastic_dNdE(I_E,rhoL_func)
        if(self.l_n2n):
            self.calc_n2n_dNdE(I_E,rhoL_func)
        if(self.l_inelastic):
            self.calc_station_inelastic_dNdE(I_E,rhoL_func)

    # Spectrum produced by scattering of incoming isotropic neutron source I_E with normalised areal density asymmetry rhoR_asym_func
    def calc_station_elastic_dNdE(self,I_E,rhoL_func):
        rhoL_asym = rhoL_func(self.elastic_mu0)
        self.elastic_dNdE = np.trapz(self.elastic_dNdEdmu*rhoL_asym*I_E[None,:],self.Ein,axis=1)

    def calc_station_inelastic_dNdE(self,I_E,rhoL_func):
        self.inelastic_dNdE = np.zeros(self.Eout.shape[0])
        for i_inelastic in range(self.n_inelastic):
            rhoL_asym = rhoL_func(self.inelastic_mu0[i_inelastic])
            self.inelastic_dNdE += np.trapz(self.inelastic_dNdEdmu[i_inelastic]*rhoL_asym*I_E[None,:],self.Ein,axis=1)

    def calc_n2n_dNdE(self,I_E,rhoL_func):
        rhoL_asym = rhoL_func(self.n2n_mu)
        grid_dNdE = np.trapz(self.n2n_ddx.rgrid*rhoL_asym[None,:,None],self.n2n_mu,axis=1)
        self.n2n_dNdE = np.trapz(I_E[:,None]*grid_dNdE,self.Ein,axis=0)

    def rhoR_2_A1s(self,rhoR):
        mbar  = self.A*Mn_kg
        A_1S = rhoR*(sigmabarn/mbar)
        return A_1S

    # # Spectrum produced by scattering of incoming neutron source with anisotropic birth spectrum
    # def elastic_scatter_aniso(self,Eout,Ein,mean_iso,mean_aniso,var_iso,b_spec,rhoR_asym_func):
    #     Ei,Eo  = np.meshgrid(Ein,Eout)
    #     muc    = col.muc(self.A,Ei,Eo,1.0,-1.0,0.0)
    #     sigma  = sigma_nT(Ein)
    #     E_vec  = Ein
    #     Tlcoeff,Nl     = interp_Tlcoeff(self.legendre_dx_spline,E_vec)
    #     Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    #     mu0 = col.mu_out(self.A,Ei,Eo,0.0)
    #     rhoR_asym = rhoR_asym_func(mu0)
    #     prim_mean = mean_iso+mean_aniso*mu0
    #     I_E_aniso = b_spec(Ei,prim_mean,var_iso)
    #     dsdO = diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
    #     jacob = col.g(self.A,Ei,Eo,1.0,-1.0,0.0)
    #     res = np.trapz(jacob*dsdO*I_E_aniso*rhoR_asym,Ein,axis=-1)
    #     return res

    #####################################################
    # Inclusion of ion velocities to scattering kernels #
    #####################################################
    def full_scattering_matrix_create(self,vvec):
        self.vvec = vvec

        Eo,vv,Ei  = np.meshgrid(self.Eout,vvec,self.Ein,indexing='ij')
        # Reverse velocity direction so +ve vf is implosion
        # Choose this way round so vf is +ve if shell coming TOWARDS detector
        vf    = -vv
        muout = col.mu_out(self.A,Ei,Eo,vf)
        jacob = col.g(self.A,Ei,Eo,1.0,muout,vf)
        flux_change = col.flux_change(Ei,1.0,vf)
        # Integrand of Eq. 8 in A. J. Crilly 2019 PoP
        dsigdOmega = xs.dsigdOmega(self.A,Ei,Eo,self.Ein,1.0,muout,vf,self)

        self.full_scattering_M  = flux_change*dsigdOmega*jacob
        self.full_scattering_mu = muout
        self.rhoL_mult          = np.ones_like(muout)

    def scattering_matrix_apply_rhoLfunc(self,rhoL_func):
        # Find multiplicative factor for areal density asymmetries
        self.rhoL_mult  = rhoL_func(self.full_scattering_mu)

    # Integrate out the birth neutron spectrum
    def matrix_primspec_int(self,I_E):
        self.M_prim = np.trapz(self.rhoL_mult*self.full_scattering_M*I_E[None,None,:],self.Ein,axis=2)

    # Integrate out the ion velocity distribution
    def matrix_interpolate_gaussian(self,E,vbar,dv):
        # Integrating over Gaussian
        gauss = np.exp(-(self.vvec-vbar)**2/2.0/(dv**2))/np.sqrt(2*np.pi)/dv
        M_v   = np.trapz(self.M_prim*gauss[None,:],self.vvec,axis=1)
        # Interpolate to energy points E
        interp = interpolate_1d(self.Eout,M_v,method='linear',bounds_error=False)
        return interp(E)

# Load in TT spectrum
# Based on Appelbe, stationary emitter, temperature range between 1 and 10 keV
# https://www.sciencedirect.com/science/article/pii/S1574181816300295
# N.B. requires some unit conversion to uniform eV
TT_data      = np.loadtxt(data_dir + "TT_spec_temprange.txt")
TT_spec_E    = TT_data[:,0]*1e6             # MeV to eV
TT_spec_T    = np.linspace(1.0,20.0,40)*1e3 # keV to eV
TT_spec_dNdE = TT_data[:,1:]/1e6            # 1/MeV to 1/eV
TT_2dinterp  = interpolate_2d(TT_spec_E,TT_spec_T,TT_spec_dNdE,method='linear',bounds_error=False,fill_value=0.0)

# TT reactivity
TT_reac_McNally_data = np.loadtxt(data_dir + "TT_reac_McNally.dat")  # sigmav im m^3/s   # From https://www.osti.gov/servlets/purl/5992170 - N.B. not in agreement with experimental measurements
TT_reac_McNally_spline = interpolate_1d(TT_reac_McNally_data[:,0],TT_reac_McNally_data[:,1],method='linear',bounds_error=False,fill_value=0.0)
TT_reac_Hale_data = np.loadtxt(data_dir + "TT_reac_Hale.dat")       # T in MeV, sigmav im cm^3/s   # From Hale
TT_reac_Hale_spline = interpolate_1d(TT_reac_Hale_data[:,0]*1e3,TT_reac_Hale_data[:,1]*1e-6,method='linear',bounds_error=False,fill_value=0.0)
# TT_reac_data = np.loadtxt(data_dir + "TT_reac_ENDF.dat")       # sigmav im m^3/s   # From ENDF
# TT_reac_spline = interpolate_1d(TT_reac_data[:,0],TT_reac_data[:,1],method='linear',bounds_error=False,fill_value=0.0)

########################
# Primary reactivities #
########################

# References:
# Bosch Hale: https://doi.org/10.1088/0029-5515/33/12/513
# Caughlan & Fowler: https://doi.org/10.1146/annurev.aa.13.090175.000441
# McNally: https://www.osti.gov/servlets/purl/5992170

# Output in m3/s, Ti in eV
def reac_DT(Ti,model='BoschHale'):
    Ti_kev = Ti/1e3
    if(model == 'BoschHale'):
        # Bosch Hale DT and DD reactivities
        # Taken from Atzeni & Meyer ter Vehn page 19
        C1 = 643.41e-22
        xi = 6.6610*Ti_kev**(-0.333333333)
        eta = 1-np.polyval([-0.10675e-3,4.6064e-3,15.136e-3,0.0e0],Ti_kev)/np.polyval([0.01366e-3,13.5e-3,75.189e-3,1.0e0],Ti_kev)
        return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)
    elif(model == 'CaughlanFowler'):
        T9 = (Ti_kev*sc.e*1e3/sc.k)/1e9
        T9_1third = T9**(1./3.)
        poly = np.polyval([17.24,10.52,1.16,1.80,0.092,1.0],T9_1third)
        return (1/sc.N_A)*(8.09e4*poly*np.exp(-4.524/T9**(1./3.)-(T9/0.120)**2) + 8.73e2*np.exp(-0.523/T9))/T9**(2./3.)
    else:
        print(f'WARNING: DT model name ({model}) not recognised! Default to 0')
        return np.zeros_like(Ti)
    
def reac_DD(Ti,model='BoschHale'):
    Ti_kev = Ti/1e3
    if(model == 'BoschHale'):
        # Bosch Hale DT and DD reactivities
        # Taken from Atzeni & Meyer ter Vehn page 19
        C1 = 3.5741e-22
        xi = 6.2696*Ti_kev**(-0.333333333)
        eta = 1-np.polyval([5.8577e-3,0.0e0],Ti_kev)/np.polyval([-0.002964e-3,7.6822e-3,1.0e0],Ti_kev)
        return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)
    elif(model == 'CaughlanFowler'):
        T9 = (Ti_kev*sc.e*1e3/sc.k)/1e9
        T9_1third = T9**(1./3.)
        poly = np.polyval([-0.071,-0.041,0.6,0.876,0.098,1.0],T9_1third)
        return (1/sc.N_A)*3.97e2/T9**(2./3.)*np.exp(-4.258/T9**(1./3.))*poly
    else:
        print(f'WARNING: DD model name ({model}) not recognised! Default to 0')
        return np.zeros_like(Ti)
    
def reac_TT(Ti,model='Hale'):
    Ti_kev = Ti/1e3
    if(model == 'Hale'):
        return TT_reac_Hale_spline(Ti_kev)
    elif(model == 'McNally'):
        return TT_reac_McNally_spline(Ti_kev)
    elif(model == 'CaughlanFowler'):
        T9 = (Ti_kev*sc.e*1e3/sc.k)/1e9
        T9_1third = T9**(1./3.)
        poly = np.polyval([0.225,0.148,-0.272,-0.455,0.086,1.0],T9_1third)
        return (1/sc.N_A)*1.67e3/T9**(2./3.)*np.exp(-4.872/T9**(1./3.))*poly
    else:
        print(f'WARNING: TT model name ({model}) not recognised! Default to 0')
        return np.zeros_like(Ti)