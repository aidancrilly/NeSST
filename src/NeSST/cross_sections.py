# Backend of spectral model
import numpy as np
from numpy.polynomial.legendre import legval
import os
from NeSST.constants import *
from NeSST.utils import *
import NeSST.collisions as col

#################################
# Loading in cross section data #
#################################

package_directory = os.path.dirname(os.path.abspath(__file__))
xsec_dir = os.path.join(package_directory,"./data/")

###############################
# Differential cross sections #
###############################

# Elastic single differential cross sections

# Interpolate the legendre coefficients (a_l) of the differential cross section
# See https://t2.lanl.gov/nis/endf/intro20.html
def interp_Tlcoeff(dx_spline,E_vec):
    size = [E_vec.shape[0]]
    NTl = len(dx_spline)
    size.append(NTl)
    Tlcoeff = np.zeros(size)
    for i in range(NTl):
        Tlcoeff[:,i] = dx_spline[i](E_vec)
    return Tlcoeff,NTl

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
def f_dsdO(Ein_vec,mu,material):

    dx_spline = material.dx_spline
    Tlcoeff_interp,Nl = interp_Tlcoeff(dx_spline,Ein_vec)
    Tlcoeff_interp    = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff_interp

    sig = material.sigma(Ein_vec)

    dsdO = diffxsec_legendre_eval(sig,mu,Tlcoeff_interp)
    return dsdO

# Differential cross section even larger wrapper function
def dsigdOmega(A,Ein,Eout,Ein_vec,muin,muout,vf,material):
    mu_CoM = col.muc(A,Ein,Eout,muin,muout,vf)
    return f_dsdO(Ein_vec,mu_CoM,material)

# Inelastic double differential cross sections

def ENDF_format(x):
    x_after_decimal = x.split('.')[1]
    if('+' in x_after_decimal):
        sign = '+'
    elif('-' in x_after_decimal):
        sign = '-'
    exp  = x.split(sign)[-1]
    num  = x[:-(len(exp)+1)]
    if(sign == '-'):
        return float(num)*10**(-int(exp))
    elif(sign == '+'):
        return float(num)*10**(+int(exp))
    else:
        print("Strange ENDF float detected....")
        print(x)
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
        self.xsec_interp = interpolate_1d(self.Ein_xsec,self.xsec,method='linear',bounds_error=False,fill_value=0.0)

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
                        self.f_ddx_interp[(Ecounter-1,Ccounter-1)] = interpolate_1d(self.Eout_ddx[(Ecounter-1,Ccounter-1)],self.f_ddx[(Ecounter-1,Ccounter-1)],method='linear',bounds_error=False,fill_value=0.0)
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
            self.f_ddx_interp[(0,i)] = interpolate_1d(self.Eout_ddx[(0,i)],self.f_ddx[(0,i)],method='linear',bounds_error=False,fill_value=0.0)
            self.Emax_ddx[(0,i)] = np.max(self.Eout_ddx[(0,i)])
            self.f_ddx_interp[(1,i)] = interpolate_1d(self.Eout_ddx[(1,i)],self.f_ddx[(1,i)],method='linear',bounds_error=False,fill_value=0.0)
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
        self.xsec_interp = interpolate_1d(self.Ein_xsec,self.xsec,method='linear',bounds_error=False,fill_value=0.0)

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