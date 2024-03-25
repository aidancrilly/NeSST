# Backend of spectral model

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

from NeSST.constants import *
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

    def __init__(self,label):
        self.label = label
        self.l_n2n = False
        self.inelastic = False
        if(self.label == 'H'):
            self.A = Mp/Mn
            elastic_xsec_file  = xs.xsec_dir + "ENDF_H1(n,elastic)_xsec.dat"
            elastic_dxsec_file = xs.xsec_dir + "ENDF_H1(n,elastic)_dx.dat"

            tot_xsec_file      = xs.xsec_dir + "ENDF_H1(n,elastic)_xsec.dat"
        elif(self.label == 'D'):
            self.A = Md/Mn
            elastic_xsec_file  = xs.xsec_dir + "ENDF_H2(n,elastic)_xsec.dat"
            elastic_dxsec_file = xs.xsec_dir + "ENDF_H2(n,elastic)_dx.dat"

            self.l_n2n         = True
            # n2n_type = 0 is ENDF LAW=6, n2n_type = 1 is tabulated double differential cross sections
            n2n_type           = 1
            n2n_xsec_file      = xs.xsec_dir + "CENDL_d(n,2n)_xsec.dat"
            n2n_dxsec_file     = xs.xsec_dir + "CENDL_d(n,2n)_ddx.dat"
            n2n_params         = None

            tot_xsec_file      = xs.xsec_dir + "ENDF_nH2_totxsec.dat"
        elif(self.label == 'T'):
            self.A = Mt/Mn
            elastic_xsec_file  = xs.xsec_dir + "ENDF_H3(n,elastic)_xsec.dat"
            elastic_dxsec_file = xs.xsec_dir + "ENDF_H3(n,elastic)_dx.dat"

            self.l_n2n         = True
            # n2n_type = 0 is ENDF LAW=6, n2n_type = 1 is tabulated double differential cross sections
            n2n_type           = 0
            n2n_xsec_file      = xs.xsec_dir + "ENDF_t(n,2n)_xsec.dat"
            n2n_dxsec_file     = None
            n2n_params         = [1.0e0,1.0e0,2.990140e0,1.0e0,3.996800e0,-6.25756e6]

            tot_xsec_file      = xs.xsec_dir + "ENDF_nH3_totxsec.dat"
        elif(self.label == '12C'):
            self.A = MC/Mn
            elastic_xsec_file  = xs.xsec_dir + "CENDL_C12(n,elastic)_xsec.dat"
            elastic_dxsec_file = xs.xsec_dir + "CENDL_C12(n,elastic)_dx.dat"

            self.inelastic     = True
            self.inelastic_Q   = -4.43890e6 # eV
            inelastic_xsec_file  = xs.xsec_dir + "CENDL_C12(n,n1)_xsec.dat"
            inelastic_dxsec_file = xs.xsec_dir + "CENDL_C12(n,n1)_dx.dat"

            tot_xsec_file      = xs.xsec_dir + "CENDL_C12_totxsec.dat"
        elif(self.label == '9Be'):
            self.A = MBe/Mn
            elastic_xsec_file  = xs.xsec_dir + "ENDF_Be9(n,elastic)_xsec.dat"
            elastic_dxsec_file = xs.xsec_dir + "ENDF_Be9(n,elastic)_dx.dat"

            self.l_n2n         = True
            # n2n_type = 0 is ENDF LAW=6, n2n_type = 1 is tabulated double differential cross sections
            n2n_type           = 1
            n2n_xsec_file      = xs.xsec_dir + "ENDF_Be9(n,2n)_xsec.dat"
            n2n_dxsec_file     = xs.xsec_dir + "ENDF_Be9(n,2n)_ddx.dat"
            n2n_params         = None

            tot_xsec_file      = xs.xsec_dir + "ENDF_nBe9_totxsec.dat"
        else:
            print("Material label "+self.label+" not recognised")

        elastic_xsec_data = self.read_ENDF_xsec_file(elastic_xsec_file)
        self.sigma = interp1d(elastic_xsec_data[:,0],elastic_xsec_data[:,1],kind='linear',bounds_error=False,fill_value=0.0)

        dx_data = np.loadtxt(elastic_dxsec_file,skiprows = 6)
        self.dx_spline = [unity]
        for i in range(1,dx_data.shape[1]):
            self.dx_spline.append(interp1d(dx_data[:,0],dx_data[:,i],kind='linear',bounds_error=False,fill_value=0.0))

        tot_xsec_data = self.read_ENDF_xsec_file(tot_xsec_file)
        self.sigma_tot = interp1d(tot_xsec_data[:,0],tot_xsec_data[:,1],kind='linear',bounds_error=False,fill_value=0.0)

        if(self.l_n2n):
            if(n2n_type == 0):
                self.n2n_ddx = xs.doubledifferentialcrosssection_LAW6(n2n_xsec_file,*n2n_params)
            elif(n2n_type == 1):
                self.n2n_ddx = xs.doubledifferentialcrosssection_data(n2n_xsec_file,n2n_dxsec_file,True)

        if(self.inelastic):
            inelastic_xsec_data = self.read_ENDF_xsec_file(inelastic_xsec_file)
            self.isigma = interp1d(inelastic_xsec_data[:,0],inelastic_xsec_data[:,1],kind='linear',bounds_error=False,fill_value=0.0)
            idx_data = np.loadtxt(inelastic_dxsec_file,skiprows = 6)
            self.idx_spline = [unity]
            for i in range(1,idx_data.shape[1]):
                self.idx_spline.append(interp1d(idx_data[:,0],idx_data[:,i],kind='linear',bounds_error=False,fill_value=0.0))

        self.Ein  = None       
        self.Eout = None
        self.vvec = None
        
    def read_ENDF_xsec_file(self,xsec_file):
        with open(xsec_file,"r") as f:
            file = f.read()
            # Read number of points
            NEin_xsec = int(file.split()[0])
            elastic_xsec_data = np.zeros((NEin_xsec,2))
            data = "".join(file.split("\n")[5:]).split()
            E = data[::2]
            x = data[1::2]
            for i in range(NEin_xsec):
                elastic_xsec_data[i,0] = xs.ENDF_format(E[i])
                elastic_xsec_data[i,1] = xs.ENDF_format(x[i])
        return elastic_xsec_data

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
        if(self.inelastic):
            self.init_station_inelastic_scatter()

    # Elastic scatter matrix
    def init_station_elastic_scatter(self):
        Ei,Eo  = np.meshgrid(self.Ein,self.Eout)
        muc    = col.muc(self.A,Ei,Eo,1.0,-1.0,0.0)
        sigma  = self.sigma(self.Ein)
        Tlcoeff,Nl     = xs.interp_Tlcoeff(self.dx_spline,self.Ein)
        Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
        self.elastic_mu0 = col.mu_out(self.A,Ei,Eo,0.0)
        dsdO = xs.diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
        jacob = col.g(self.A,Ei,Eo,1.0,-1.0,0.0)
        self.elastic_dNdEdmu = jacob*dsdO

    # Inelastic scatter matrix
    # Currently uses classical kinematics
    def init_station_inelastic_scatter(self):
        Ei,Eo  = np.meshgrid(self.Ein,self.Eout)
        kin_a  = np.sqrt((self.A/(self.A+1))**2*(1.0+(self.A+1)/self.A*self.inelastic_Q/Ei))
        kin_b  = 1.0/(self.A+1)
        muc    = ((Eo/Ei)-kin_a**2-kin_b**2)/(2*kin_a*kin_b)
        sigma  = self.isigma(self.Ein)
        Tlcoeff,Nl     = xs.interp_Tlcoeff(self.idx_spline,self.Ein)
        Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
        self.inelastic_mu0 = (np.sqrt(Eo/Ei)-(kin_a**2-kin_b**2)*np.sqrt(Ei/Eo))/(2*kin_b)
        dsdO = xs.diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
        jacob = 2.0/((kin_a+kin_b)**2-(kin_a-kin_b)**2)/Ei
        self.inelastic_dNdEdmu = jacob*dsdO

    def init_n2n_ddxs(self,Nm=100):
        self.n2n_mu = np.linspace(-1.0,1.0,Nm)
        self.n2n_ddx.regular_grid(self.Ein,self.n2n_mu,self.Eout)

    def calc_dNdEs(self,I_E,rhoL_func):
        self.calc_station_elastic_dNdE(I_E,rhoL_func)
        if(self.l_n2n):
            self.calc_n2n_dNdE(I_E,rhoL_func)
        if(self.inelastic):
            self.calc_station_inelastic_dNdE(I_E,rhoL_func)

    # Spectrum produced by scattering of incoming isotropic neutron source I_E with normalised areal density asymmetry rhoR_asym_func
    def calc_station_elastic_dNdE(self,I_E,rhoL_func):
        rhoL_asym = rhoL_func(self.elastic_mu0)
        self.elastic_dNdE = np.trapz(self.elastic_dNdEdmu*rhoL_asym*I_E[None,:],self.Ein,axis=1)

    def calc_station_inelastic_dNdE(self,I_E,rhoL_func):
        rhoL_asym = rhoL_func(self.elastic_mu0)
        self.inelastic_dNdE = np.trapz(self.inelastic_dNdEdmu*rhoL_asym*I_E[None,:],self.Ein,axis=1)

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
    #     Tlcoeff,Nl     = interp_Tlcoeff(self.dx_spline,E_vec)
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
        interp = interp1d(self.Eout,M_v,kind='linear',copy=False,bounds_error=False)
        return interp(E)


# Default load
mat_H = material_data('H')
mat_D = material_data('D')
mat_T = material_data('T')
mat_12C = material_data('12C')
mat_9Be = material_data('9Be')

available_materials_dict = {"H" : mat_H, "D" : mat_D, "T" : mat_T, "12C" : mat_12C, "9Be" : mat_9Be}

# Load in TT spectrum
# Based on Appelbe, stationary emitter, temperature range between 1 and 10 keV
# https://www.sciencedirect.com/science/article/pii/S1574181816300295
# N.B. requires some unit conversion to uniform eV
TT_data      = np.loadtxt(xs.xsec_dir + "TT_spec_temprange.txt")
TT_spec_E    = TT_data[:,0]*1e6             # MeV to eV
TT_spec_T    = np.linspace(1.0,20.0,40)*1e3 # keV to eV
TT_spec_dNdE = TT_data[:,1:]/1e6            # 1/MeV to 1/eV
TT_2dinterp  = interp2d(TT_spec_E,TT_spec_T,TT_spec_dNdE.T,kind='linear',bounds_error=False,fill_value=0.0)

# TT reactivity
# TT_reac_data = np.loadtxt(xs.xsec_dir + "TT_reac_McNally.dat")  # sigmav im m^3/s   # From https://www.osti.gov/servlets/purl/5992170 - N.B. not in agreement with experimental measurements
TT_reac_data = np.loadtxt(xs.xsec_dir + "TT_reac_ENDF.dat")       # sigmav im m^3/s   # From ENDF
TT_reac_spline = interp1d(TT_reac_data[:,0],TT_reac_data[:,1],kind='cubic',bounds_error=False,fill_value=0.0)

########################
# Primary reactivities #
########################

# Bosh Hale DT and DD reactivities
# Taken from Atzeni & Meyer ter Vehn page 19
# Output in m3/s, Ti in eV
def reac_DT(Ti):
    Ti_kev = Ti/1e3
    C1 = 643.41e-22
    xi = 6.6610*Ti_kev**(-0.333333333)
    eta = 1-np.polyval([-0.10675e-3,4.6064e-3,15.136e-3,0.0e0],Ti_kev)/np.polyval([0.01366e-3,13.5e-3,75.189e-3,1.0e0],Ti_kev)
    return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)

def reac_DD(Ti):
    Ti_kev = Ti/1e3
    C1 = 3.5741e-22
    xi = 6.2696*Ti_kev**(-0.333333333)
    eta = 1-np.polyval([5.8577e-3,0.0e0],Ti_kev)/np.polyval([-0.002964e-3,7.6822e-3,1.0e0],Ti_kev)
    return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)

def reac_TT(Ti):
    Ti_kev = Ti/1e3
    return TT_reac_spline(Ti_kev)

##############################################################################
# Deprecated n2n matrix representation
E1_n2n = np.linspace(13.0e6,15.0e6,100)
E2_n2n = np.linspace(1.0e6,13.0e6,500)
Dn2n_matrix = np.loadtxt(xs.xsec_dir + "Dn2n_matrix.dat")
Tn2n_matrix_1 = np.loadtxt(xs.xsec_dir + "Tn2n_matrix_ENDFLAW6.dat")
Tn2n_matrix_2 = np.loadtxt(xs.xsec_dir + "Tn2n_matrix_CENDL_transform.dat")
# 2D interpolation functions
Dn2n_2dinterp = interp2d(E1_n2n,E2_n2n,Dn2n_matrix.T,kind='linear',bounds_error=False,fill_value=0.0)
Tn2n_1_2dinterp = interp2d(E1_n2n,E2_n2n,Tn2n_matrix_1.T,kind='linear',bounds_error=False,fill_value=0.0)
Tn2n_2_2dinterp = interp2d(E1_n2n,E2_n2n,Tn2n_matrix_2.T,kind='linear',bounds_error=False,fill_value=0.0)
# Deprecated n2n matrix representation
############################################################################