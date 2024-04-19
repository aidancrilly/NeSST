from .constants import *
from .stopping_power import *
import NeSST.cross_sections as xs

def mat_knockon_ion_spectrum(mat,Ein,Iin,muc,normalise=False):

    A = mat.A

    a_2 = A/(A+1)**2
    b_2 = A/(A+1)**2
    a = np.sqrt(a_2)
    b = np.sqrt(b_2)

    x_n            = a_2+b_2+2*a*b*muc
    E_p            = x_n*np.amax(Ein)
    sigma          = mat.sigma(Ein)
    Ei,mu          = np.meshgrid(Ein,-muc)

    Tlcoeff,Nl     = xs.interp_Tlcoeff(mat.legendre_dx_spline,Ein)
    Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    dsdO_n         = xs.diffxsec_legendre_eval(sigma,mu,Tlcoeff_interp)
    g_n            = 2.0/((a+b)**2-(a-b)**2)/Ei
    dsdE_p         = np.trapz(g_n*dsdO_n*Iin,Ein,axis=-1)

    if(normalise):
        dsdE_p /= np.trapz(dsdE_p,x=E_p)

    return E_p,dsdE_p


def local_stopping_model(Zp,mp,Ep,dNdEp,ne,Te):
    # See Hayes et al. equation 1 https://aip.scitation.org/doi/pdf/10.1063/1.4928104
    return np.cumsum(dNdEp[::-1])[::-1]/(dEdx_electron_MaynardDeutsch(Zp,mp,Ep,ne,Te)/(ne/1e32))

def nonlocal_stopping_model(Zp,mp,Ep,dNdEp,ne,Te,L):
    beta = dEdx_electron_MaynardDeutsch(Zp,mp,Ep,ne,Te)/(ne/1e32)
    betaL = L*dEdx_electron_MaynardDeutsch(Zp,mp,Ep,ne,Te)
    dNdEo = np.zeros_like(dNdEp)
    dE = Ep[0]-Ep[1]
    for i in range(len(dNdEo)-2,0,-1):
        dNdEo[i] = dNdEo[i+1]*(1.0+dE/betaL[i])-dNdEp[i]*dE
    psi = dNdEo/beta
    return psi