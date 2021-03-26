# Backend of spectral model

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import os
import collisions as col

#################################
# Loading in cross section data #
#################################

package_directory = os.path.dirname(os.path.abspath(__file__))
xsec_dir = os.path.join(package_directory,"../data/")

###############################
# Differential cross sections #
###############################

# Elastic single differential cross sections

# Interpolate the legendre coefficients (a_l) of the differential cross section
# See https://t2.lanl.gov/nis/endf/intro20.html
def interp_Tlcoeff(reac_type,E_vec):
    size = [E_vec.shape[0]]
    if(reac_type == "nD"):
        NTl = len(nD_dx_spline)
        size.append(NTl)
        Tlcoeff = np.zeros(size)
        for i in range(NTl):
            Tlcoeff[:,i] = nD_dx_spline[i](E_vec)
    elif(reac_type == "nT"):
        NTl = len(nT_dx_spline)
        size.append(NTl)
        Tlcoeff = np.zeros(size)
        for i in range(NTl):
            Tlcoeff[:,i] = nT_dx_spline[i](E_vec)
    else:
        print('WARNING: reac_type != nD or nT in interp_Tlcoeff function')
    return Tlcoeff,NTl

# Cross section
def sigma(Ein_vec,reac_type):
    if(reac_type == "nD"):
        return sigma_nD(Ein_vec)
    elif(reac_type == "nT"):
        return sigma_nT(Ein_vec)

# Evaluate the differential cross section by combining legendre and cross section
# See https://t2.lanl.gov/nis/endf/intro20.html
def diffxsec_legendre_eval(sig,mu,coeff):
    c = coeff.T
    ans = np.zeros_like(mu)
    if(len(mu.shape) == 1):
        ans = sig*legval(mu,c,tensor=False)
    elif(len(mu.shape) == 2):
        ans = sig*legval(mu,c[:,None,:],tensor=False)
    elif(len(mu.shape) == 3):
        ans = sig*legval(mu,c[:,None,None,:],tensor=False)
    ans[np.abs(mu) > 1.0] = 0.0
    return ans

# CoM frame differential cross section wrapper fucntion
def f_dsdO(Ein_vec,mu,reac_type):
    E_vec = 1e6*Ein_vec
    NE    = len(E_vec)
    Tlcoeff_interp,Nl = interp_Tlcoeff(reac_type,E_vec)
    Tlcoeff_interp    = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff_interp
    sig = sigma(Ein_vec,reac_type)
    dsdO = diffxsec_legendre_eval(sig,mu,Tlcoeff_interp)
    return dsdO

# Differential cross section even larger wrapper function
def dsigdOmega(A,Ein,Eout,Ein_vec,muin,muout,vf,reac_type):
    mu_CoM = col.muc(A,Ein,Eout,muin,muout,vf)
    return f_dsdO(Ein_vec,mu_CoM,reac_type)

# Inelastic double differential cross sections

def ENDF_format(x):
    sign = x[-2]
    exp  = x[-1]
    num  = x[:-2]
    if(sign == '-'):
        return float(num)*10**(-int(exp))
    elif(sign == '+'):
        return float(num)*10**(+int(exp))
    else:
        print("Strange ENDF float detected....")
        return 0.0

# Reads and interpolated data saved in the ENDF interpreted data format
class doubledifferentialcrosssection_data:

    def __init__(self,filexsec,fileddx,ENDF):
        self.filexsec = filexsec
        self.fileddx = fileddx
        self.ENDF = ENDF
        if(ENDF):
            self.read_xsec_file()
            self.read_ddx_file()
        else:
            self.read_ddx_file_csv()

    def read_xsec_file(self):
        with open(self.filexsec,"r") as f:
            file = f.read()
            # Read number of points
            self.NEin_xsec = int(file.split()[0])
            self.Ein_xsec  = np.zeros(self.NEin_xsec)
            self.xsec      = np.zeros(self.NEin_xsec)
            counter        = 0
            data = "".join(file.split("\n")[5:]).split()
            E = data[::2]
            x = data[1::2]
            for i in range(self.NEin_xsec):
                self.Ein_xsec[i] = ENDF_format(E[i])/1e6
                self.xsec[i]     = ENDF_format(x[i])
        self.xsec_interp = interp1d(self.Ein_xsec,self.xsec,kind='linear',bounds_error=False,fill_value=0.0)

    def read_ddx_file(self):
        with open(self.fileddx,"r") as f:
            file = f.read().split("\n")
            # Read number of points
            self.NEin_ddx = int(file[1].split()[0])
            self.Ein_ddx  = np.zeros(self.NEin_ddx)
            # Read data
            Ecounter = 0
            Ccounter = 0
            read_cosine = False
            read_energy = False
            read_data   = False
            self.Ncos_ddx  = []
            self.cos_ddx   = []
            self.NEout_ddx = {}
            self.Eout_ddx  = {}
            self.f_ddx     = {}
            self.f_ddx_interp = {}
            self.Emax_ddx     = {}
            # Read in all the array axes
            for Lcounter,line in enumerate(file):
                line_split = line.split()
                if(line_split != []):
                    # Read number of cosines for given incoming energy
                    if(read_cosine):
                        self.Ncos_ddx.append(int(line_split[0]))
                        self.cos_ddx.append(np.zeros(int(line_split[0])))
                        read_cosine = False
                        Ccounter = 0
                    # Read number of energies for given incoming energy and cosine
                    if(read_energy):
                        NEout = int(line_split[0])
                        self.NEout_ddx[(Ecounter-1,Ccounter-1)] = NEout
                        self.Eout_ddx[(Ecounter-1,Ccounter-1)]  = np.zeros(NEout)
                        self.f_ddx[(Ecounter-1,Ccounter-1)]     = np.zeros(NEout)
                        read_energy = False
                        idx1 = Lcounter + 4
                        idx2 = idx1 + int(np.ceil(NEout/3)) + 1
                        read_data   = True
                    # Read in the data
                    if(read_data):
                        data = "".join(file[idx1:idx2]).split()
                        E = data[::2]
                        x = data[1::2]
                        for i in range(NEout):
                            self.Eout_ddx[(Ecounter-1,Ccounter-1)][i] = ENDF_format(E[i])/1e6
                            self.f_ddx[(Ecounter-1,Ccounter-1)][i]    = ENDF_format(x[i])*1e6
                        self.f_ddx_interp[(Ecounter-1,Ccounter-1)] = interp1d(self.Eout_ddx[(Ecounter-1,Ccounter-1)],self.f_ddx[(Ecounter-1,Ccounter-1)],kind='linear',bounds_error=False,fill_value=0.0)
                        self.Emax_ddx[(Ecounter-1,Ccounter-1)] = np.max(self.Eout_ddx[(Ecounter-1,Ccounter-1)])
                        read_data = False
                    # Read incoming energy
                    if(line_split[0] == 'Energy:'):
                        self.Ein_ddx[Ecounter] = ENDF_format(line_split[1])/1e6
                        Ecounter += 1
                    # Prep for cosine number read in
                    elif(" ".join(line_split) == 'Cosine Interpolation:'):
                        read_cosine = True
                    # Read number of secondary energies
                    elif(" ".join(line_split) == 'Secondary-Energy Interpolation:'):
                        read_energy = True
                    elif('Cosine:' in line):
                        line_split_c = line.split(":")[1]
                        self.cos_ddx[Ecounter-1][Ccounter] = ENDF_format(line_split_c)
                        Ccounter += 1

    def read_ddx_file_csv(self):
        self.NEin_ddx = 2
        self.Ein_ddx  = np.array([0.0,14.0])
        data = np.loadtxt(self.fileddx,delimiter=',',skiprows=1)
        angles,counts = np.unique(data[:,-1],return_counts=True)
        cos = np.cos(angles[::-1]*np.pi/180.)
        NC = cos.shape[0]
        self.Ncos_ddx  = [NC,NC]
        self.cos_ddx   = [cos,cos]
        E_prev = 0.0
        self.NEout_ddx = {}
        self.Eout_ddx  = {}
        self.f_ddx     = {}
        self.f_ddx_interp = {}
        self.Emax_ddx     = {}
        idx = data[:,0].shape[0]
        i = 0
        for ic in range(NC-1,-1,-1):
            NEout = counts[ic]
            self.NEout_ddx[(0,i)] = NEout
            self.Eout_ddx[(0,i)]  = data[idx-NEout:idx,1]
            self.f_ddx[(0,i)]     = np.zeros(NEout)
            self.NEout_ddx[(1,i)] = NEout
            self.Eout_ddx[(1,i)]  = data[idx-NEout:idx,1]
            # From barns to mbarns, from sr to per cosine, from number of neutrons to cross section
            self.f_ddx[(1,i)]     = 0.5*(2*np.pi)*data[idx-NEout:idx,0]/1e3
            self.f_ddx_interp[(0,i)] = interp1d(self.Eout_ddx[(0,i)],self.f_ddx[(0,i)],kind='linear',bounds_error=False,fill_value=0.0)
            self.Emax_ddx[(0,i)] = np.max(self.Eout_ddx[(0,i)])
            self.f_ddx_interp[(1,i)] = interp1d(self.Eout_ddx[(1,i)],self.f_ddx[(1,i)],kind='linear',bounds_error=False,fill_value=0.0)
            self.Emax_ddx[(1,i)] = np.max(self.Eout_ddx[(1,i)])
            idx -= NEout
            i   += 1
        self.xsec_interp = lambda x : 1.

    # Interpolate using Unit Base Transform
    def interpolate(self,Ein,mu,Eout):

        # Find indices
        # Energies
        if(Ein == np.amax(self.Ein_ddx)):
            Eidx2 = self.NEin_ddx - 1
            Eidx1 = Eidx2 - 1
        elif(Ein < np.amin(self.Ein_ddx)):
            return 0.0
        else:
            Eidx2 = np.argmax(self.Ein_ddx > Ein)
            Eidx1 = Eidx2 - 1
        # Angles
        if(mu == +1.0):
            Cidx12 = self.Ncos_ddx[Eidx1]-1
            Cidx11 = Cidx12 - 1
            Cidx22 = self.Ncos_ddx[Eidx2]-1
            Cidx21 = Cidx22 - 1
        elif(mu == -1.0):
            Cidx12 = 1
            Cidx11 = 0
            Cidx22 = 1
            Cidx21 = 0
        else:
            Cidx12 = np.argmax(self.cos_ddx[Eidx1] > mu)
            Cidx11 = Cidx12 - 1
            Cidx22 = np.argmax(self.cos_ddx[Eidx2] > mu)
            Cidx21 = Cidx22 - 1

        # Find interpolation factors
        mu_x1 = (mu-self.cos_ddx[Eidx1][Cidx11])/(self.cos_ddx[Eidx1][Cidx12]-self.cos_ddx[Eidx1][Cidx11])
        mu_x2 = (mu-self.cos_ddx[Eidx2][Cidx21])/(self.cos_ddx[Eidx2][Cidx22]-self.cos_ddx[Eidx2][Cidx21])
        Ein_x = (Ein-self.Ein_ddx[Eidx1])/(self.Ein_ddx[Eidx2]-self.Ein_ddx[Eidx1])

        x_112 = mu_x1
        x_111 = (1-x_112)
        x_222 = mu_x2
        x_221 = (1-x_222)

        x_2 = Ein_x
        x_1 = (1-x_2)

        # Unit base transform
        E_h11 = self.Emax_ddx[(Eidx1,Cidx11)]
        E_h12 = self.Emax_ddx[(Eidx1,Cidx12)]
        E_h21 = self.Emax_ddx[(Eidx2,Cidx21)]
        E_h22 = self.Emax_ddx[(Eidx2,Cidx22)]
        E_h1  = E_h11 + mu_x1*(E_h12-E_h11)
        E_h2  = E_h21 + mu_x2*(E_h22-E_h21)
        E_high = E_h1 + Ein_x*(E_h2-E_h1)
        if(E_high == 0.0):
            return 0.0
        J_111 = self.Emax_ddx[(Eidx1,Cidx11)]/E_high
        J_112 = self.Emax_ddx[(Eidx1,Cidx12)]/E_high
        J_221 = self.Emax_ddx[(Eidx2,Cidx21)]/E_high
        J_222 = self.Emax_ddx[(Eidx2,Cidx22)]/E_high

        # Find unit base transformed energy
        Eout_111 = Eout*J_111
        Eout_112 = Eout*J_112
        Eout_221 = Eout*J_221
        Eout_222 = Eout*J_222

        f_111 = self.f_ddx_interp[(Eidx1,Cidx11)](Eout_111)*J_111
        f_112 = self.f_ddx_interp[(Eidx1,Cidx12)](Eout_112)*J_112
        f_221 = self.f_ddx_interp[(Eidx2,Cidx21)](Eout_221)*J_221
        f_222 = self.f_ddx_interp[(Eidx2,Cidx22)](Eout_222)*J_222

        f_1 = x_111*f_111+x_112*f_112
        f_2 = x_221*f_221+x_222*f_222

        f_ddx = x_1*f_1+x_2*f_2

        return f_ddx

    def regular_grid(self,Ein,mu,Eout):
        self.rgrid_shape = (Ein.shape[0],mu.shape[0],Eout.shape[0])
        self.rgrid = np.zeros(self.rgrid_shape)
        for i in range(Ein.shape[0]):
            for j in range(mu.shape[0]):
                self.rgrid[i,j,:] = 2.*self.xsec_interp(Ein[i])*self.interpolate(Ein[i],mu[j],Eout)

    def ENDF_format(self,x):
        sign = x[-2]
        exp  = x[-1]
        num  = x[:-2]
        if(sign == '-'):
            return float(num)*10**(-int(exp))
        elif(sign == '+'):
            return float(num)*10**(+int(exp))
        else:
            print("Strange ENDF float detected....")
            return 0.0

class doubledifferentialcrosssection_LAW6:

    def __init__(self,filexsec,A_i,A_e,A_t,A_p,A_tot,Q_react):
        self.A_i     = A_i
        self.A_e     = A_e
        self.A_t     = A_t
        self.A_p     = A_p
        self.A_tot   = A_tot
        self.Q_react = Q_react
        self.filexsec = filexsec
        self.read_xsec_file()

    def read_xsec_file(self):
        with open(self.filexsec,"r") as f:
            file = f.read()
            # Read number of points
            self.NEin_xsec = int(file.split()[0])
            self.Ein_xsec  = np.zeros(self.NEin_xsec)
            self.xsec      = np.zeros(self.NEin_xsec)
            counter        = 0
            data = "".join(file.split("\n")[5:]).split()
            E = data[::2]
            x = data[1::2]
            for i in range(self.NEin_xsec):
                self.Ein_xsec[i] = ENDF_format(E[i])/1e6
                self.xsec[i]     = ENDF_format(x[i])
        self.xsec_interp = interp1d(self.Ein_xsec,self.xsec,kind='linear',bounds_error=False,fill_value=0.0)

    def ddx(self,Ein,mu,Eout):
        E_star = Ein*self.A_i*self.A_e/(self.A_t+self.A_i)**2
        E_a    = self.A_t*Ein/(self.A_p+self.A_t)+self.Q_react
        E_max  = (self.A_tot-1.0)*E_a/self.A_tot
        C3     = 4.0/(np.pi*E_max*E_max)
        square_bracket_term = E_max-(E_star+Eout-2*mu*np.sqrt(E_star*Eout))
        square_bracket_term[square_bracket_term < 0.0] = 0.0
        f_ddx = C3*np.sqrt(Eout*square_bracket_term)
        return f_ddx

    def regular_grid(self,Ein,mu,Eout):
        Ei,Mm,Eo = np.meshgrid(Ein,mu,Eout,indexing='ij')
        self.rgrid = 2.*self.xsec_interp(Ei)*self.ddx(Ei,Mm,Eo)

# Cross sections
nDnT_xsec_data = np.loadtxt(xsec_dir + "nDnT_xsec.dat")
sigma_nD = interp1d(nDnT_xsec_data[:,0],nDnT_xsec_data[:,1]*1e28,kind='linear',bounds_error=False,fill_value=0.0)
sigma_nT = interp1d(nDnT_xsec_data[:,0],nDnT_xsec_data[:,2]*1e28,kind='linear',bounds_error=False,fill_value=0.0)

# Differential cross sections
def unity(x):
    return np.ones_like(x)
# Elastic nD scattering
nD_dx_data = np.loadtxt(xsec_dir + "ENDF_H2(n,elastic)_dx.dat",skiprows = 6)
nD_dx_spline = [unity]
for i in range(1,nD_dx_data.shape[1]):
    nD_dx_spline.append(interp1d(nD_dx_data[:,0],nD_dx_data[:,i],kind='linear',bounds_error=False,fill_value=0.0))
# Elastic nT scattering
nT_dx_data = np.loadtxt(xsec_dir + "ENDF_H3(n,elastic)_dx.dat",skiprows = 6)
nT_dx_spline = [unity]
for i in range(1,nT_dx_data.shape[1]):
    nT_dx_spline.append(interp1d(nT_dx_data[:,0],nT_dx_data[:,i],kind='linear',bounds_error=False,fill_value=0.0))

# Total cross sections
tot_xsec_data = np.loadtxt(xsec_dir + "tot_D_xsec.dat")
sigma_D_tot = interp1d(tot_xsec_data[:,0],tot_xsec_data[:,1]*1e28,kind='linear',bounds_error=False,fill_value=0.0)
tot_xsec_data = np.loadtxt(xsec_dir + "tot_T_xsec.dat")
sigma_T_tot = interp1d(tot_xsec_data[:,0],tot_xsec_data[:,1]*1e28,kind='linear',bounds_error=False,fill_value=0.0)

Dn2n_ddx = doubledifferentialcrosssection_data(xsec_dir + "CENDL_d(n,2n)_xsec.dat",xsec_dir + "CENDL_d(n,2n)_ddx.dat",True)
Tn2n_ddx = doubledifferentialcrosssection_LAW6(xsec_dir + "ENDF_t(n,2n)_xsec.dat",1.0e0,1.0e0,2.990140e0,1.0e0,3.996800e0,-6.25756e0)

##############################################################################
# Deprecated n2n matrix representation
E1_n2n = np.linspace(13,15,100)
E2_n2n = np.linspace(1.0,13,500)
Dn2n_matrix = np.loadtxt(xsec_dir + "Dn2n_matrix.dat")
Tn2n_matrix_1 = np.loadtxt(xsec_dir + "Tn2n_matrix_ENDFLAW6.dat")
Tn2n_matrix_2 = np.loadtxt(xsec_dir + "Tn2n_matrix_CENDL_transform.dat")
# 2D interpolation functions
Dn2n_2dinterp = interp2d(E1_n2n,E2_n2n,Dn2n_matrix.T,kind='linear',bounds_error=False,fill_value=0.0)
Tn2n_1_2dinterp = interp2d(E1_n2n,E2_n2n,Tn2n_matrix_1.T,kind='linear',bounds_error=False,fill_value=0.0)
Tn2n_2_2dinterp = interp2d(E1_n2n,E2_n2n,Tn2n_matrix_2.T,kind='linear',bounds_error=False,fill_value=0.0)
# Deprecated n2n matrix representation
############################################################################

# Load in TT spectrum
# Based on Appelbe, stationary emitter, temperature range between 1 and 10 keV
# https://www.sciencedirect.com/science/article/pii/S1574181816300295
TT_data      = np.loadtxt(xsec_dir + "TT_spec_temprange.txt")
TT_spec_E    = TT_data[:,0]
TT_spec_T    = np.linspace(1.0,20.0,40)
TT_spec_dNdE = TT_data[:,1:]
TT_2dinterp  = interp2d(TT_spec_E,TT_spec_T,TT_spec_dNdE.T,kind='linear',bounds_error=False,fill_value=0.0)

# TT reactivity
# TT_reac_data = np.loadtxt(xsec_dir + "TT_reac_McNally.dat")  # sigmav im m^3/s   # From https://www.osti.gov/servlets/purl/5992170 - N.B. not in agreement with experimental measurements
TT_reac_data = np.loadtxt(xsec_dir + "TT_reac_ENDF.dat")       # sigmav im m^3/s   # From ENDF
TT_reac_spline = interp1d(TT_reac_data[:,0],TT_reac_data[:,1],kind='cubic',bounds_error=False,fill_value=0.0)

########################
# Primary reactivities #
########################

# Bosh Hale DT and DD reactivities
# Taken from Atzeni & Meyer ter Vehn page 19
# Output in m3/s, Ti in keV
def reac_DT(Ti):
    C1 = 643.41e-22
    xi = 6.6610*Ti**(-0.333333333)
    eta = 1-np.polyval([-0.10675e-3,4.6064e-3,15.136e-3,0.0e0],Ti)/np.polyval([0.01366e-3,13.5e-3,75.189e-3,1.0e0],Ti)
    return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)

def reac_DD(Ti):
    C1 = 3.5741e-22
    xi = 6.2696*Ti**(-0.333333333)
    eta = 1-np.polyval([5.8577e-3,0.0e0],Ti)/np.polyval([-0.002964e-3,7.6822e-3,1.0e0],Ti)
    return C1*eta**(-0.833333333)*xi**2*np.exp(-3*eta**(0.333333333)*xi)

def reac_TT(Ti):
    return TT_reac_spline(Ti)

############################################
# Stationary ion scattered spectral shapes #
############################################

# Spectrum produced by scattering of incoming isotropic neutron source I_E by tritium
def nTspec(Eout,Ein,I_E,A_T,P1_mag = 0.0):
    Ei,Eo  = np.meshgrid(Ein,Eout)
    muc    = col.muc(A_T,Ei,Eo,1.0,-1.0,0.0)
    sigma  = sigma_nT(Ein)
    E_vec  = 1e6*Ein
    Tlcoeff,Nl     = interp_Tlcoeff("nT",E_vec)
    Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    mu0 = col.mu_out(A_T,Ei,Eo,0.0)
    dsdO = diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
    jacob = col.g(A_T,Ei,Eo,1.0,-1.0,0.0)
    if(np.isscalar(P1_mag)):
        rhoR_asym = 1.0+P1_mag*mu0
        res = np.trapz(jacob*dsdO*I_E*rhoR_asym,Ein,axis=-1)
    else:
        rhoR_asym = 1.0+P1_mag[None,None,:]*mu0[:,:,None]
        res = np.trapz(jacob[:,:,None]*dsdO[:,:,None]*I_E[None,:,None]*rhoR_asym,Ein,axis=1)
    return res

# Spectrum produced by scattering of incoming isotropic neutron source I_E by deuterium
def nDspec(Eout,Ein,I_E,A_D,P1_mag = 0.0):
    Ei,Eo  = np.meshgrid(Ein,Eout)
    muc    = col.muc(A_D,Ei,Eo,1.0,-1.0,0.0)
    sigma  = sigma_nD(Ein)
    E_vec  = 1e6*Ein
    Tlcoeff,Nl     = interp_Tlcoeff("nD",E_vec)
    Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    mu0 = col.mu_out(A_D,Ei,Eo,0.0)
    dsdO = diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
    jacob = col.g(A_D,Ei,Eo,1.0,-1.0,0.0)
    if(np.isscalar(P1_mag)):
        rhoR_asym = 1.0+P1_mag*mu0
        res = np.trapz(jacob*dsdO*I_E*rhoR_asym,Ein,axis=-1)
    else:
        rhoR_asym = 1.0+P1_mag[None,None,:]*mu0[:,:,None]
        res = np.trapz(jacob[:,:,None]*dsdO[:,:,None]*I_E[None,:,None]*rhoR_asym,Ein,axis=1)
    return res

# nT spectrum produced by scattering of incoming neutron source with anisotropic birth spectrum
def nTspec_aniso(Eout,Ein,mean_iso,mean_aniso,var_iso,b_spec,A_T,P1_mag):
    Ei,Eo  = np.meshgrid(Ein,Eout)
    muc    = col.muc(A_T,Ei,Eo,1.0,-1.0,0.0)
    sigma  = sigma_nT(Ein)
    E_vec  = 1e6*Ein
    Tlcoeff,Nl     = interp_Tlcoeff("nT",E_vec)
    Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    mu0 = col.mu_out(A_T,Ei,Eo,0.0)
    rhoR_asym = 1.0+P1_mag*mu0
    prim_mean = mean_iso+mean_aniso*mu0
    I_E_aniso = b_spec(Ei,prim_mean,var_iso)
    dsdO = diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
    jacob = col.g(A_T,Ei,Eo,1.0,-1.0,0.0)
    res = np.trapz(jacob*dsdO*I_E_aniso*rhoR_asym,Ein,axis=-1)
    return res

# nD spectrum produced by scattering of incoming neutron source with anisotropic birth spectrum
def nDspec_aniso(Eout,Ein,mean_iso,mean_aniso,var_iso,b_spec,A_D,P1_mag):
    Ei,Eo  = np.meshgrid(Ein,Eout)
    muc    = col.muc(A_D,Ei,Eo,1.0,-1.0,0.0)
    sigma  = sigma_nD(Ein)
    E_vec  = 1e6*Ein
    Tlcoeff,Nl     = interp_Tlcoeff("nD",E_vec)
    Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    mu0 = col.mu_out(A_D,Ei,Eo,0.0)
    rhoR_asym = 1.0+P1_mag*mu0
    prim_mean = mean_iso+mean_aniso*mu0
    I_E_aniso = b_spec(Ei,prim_mean,var_iso)
    dsdO = diffxsec_legendre_eval(sigma,muc,Tlcoeff_interp)
    jacob = col.g(A_D,Ei,Eo,1.0,-1.0,0.0)
    res = np.trapz(jacob*dsdO*I_E_aniso*rhoR_asym,Ein,axis=-1)
    return res

def init_n2n_ddxs(Eout,Ein,I_E,Nm=100):
    mu = np.linspace(-1.0,1.0,Nm)
    Dn2n_ddx.regular_grid(Ein,mu,Eout)
    Tn2n_ddx.regular_grid(Ein,mu,Eout)
    Dn2n_ddx.rgrid_sym = np.trapz(Dn2n_ddx.rgrid,mu,axis=1)
    Tn2n_ddx.rgrid_sym = np.trapz(Tn2n_ddx.rgrid,mu,axis=1)
    Dn2n_ddx.dNdE_sym  = np.trapz(I_E[:,None]*Dn2n_ddx.rgrid_sym,Ein,axis=0)
    Tn2n_ddx.dNdE_sym  = np.trapz(I_E[:,None]*Tn2n_ddx.rgrid_sym,Ein,axis=0)


def init_n2n_ddxs_mode1(Eout,Ein,I_E,P1,Nm=100):
    mu = np.linspace(-1.0,1.0,Nm)
    Dn2n_ddx.regular_grid(Ein,mu,Eout)
    Tn2n_ddx.regular_grid(Ein,mu,Eout)
    Dn2n_ddx.rgrid_IE = np.trapz(I_E[:,None,None]*Dn2n_ddx.rgrid,Ein,axis=0)
    Tn2n_ddx.rgrid_IE = np.trapz(I_E[:,None,None]*Tn2n_ddx.rgrid,Ein,axis=0)
    Dn2n_ddx.rgrid_P1 = np.trapz(Dn2n_ddx.rgrid_IE[:,:,None]*(1+P1[None,None,:]*mu[:,None,None]),mu,axis=0)
    Tn2n_ddx.rgrid_P1 = np.trapz(Tn2n_ddx.rgrid_IE[:,:,None]*(1+P1[None,None,:]*mu[:,None,None]),mu,axis=0)