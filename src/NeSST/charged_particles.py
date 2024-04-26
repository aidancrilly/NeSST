from .constants import *
from .stopping_power import *
import NeSST.cross_sections as xs

def RutherfordDiffXSec(Z1,Z2,m1,m2,Ein,muc):
    # 2-body Coulomb collision in the CoM frame
    # 1 - projectile, 2 - target, in lab frame 2 is assumed stationary

    # Energy in CoM
    E_r    = Ein*m2/(m1+m2) # eV
    # Impact parameter for 90 degree scatter
    b_90   = 7.199822e-10*Z1*Z2/E_r # m
    return 2*np.pi*1e28*(b_90/(1.0e0-muc))**2+0.5 # barns/cosine

def Rutherford_knockon_ion_spectrum(mi,mj,mk,Zi,Zj,Ein,Iin,muc,normalise=False):
    # Mass conserving
    ml = (mi+mj)-mk

    mij = mi*mj/(mi+mj)
    mlk = ml*mk/(ml+mk)

    a_2 = mlk*mij/(mk*mi)
    b_2 = mij**2*mk/(mj**2*mi)
    a = np.sqrt(a_2)
    b = np.sqrt(b_2)

    Ei,mu          = np.meshgrid(Ein,-muc)
    x_n            = a_2+b_2-2*a*b*mu
    E_ps           = x_n*Ei
    E_p            = (a_2+b_2+2*a*b*muc)*np.amax(Ein)
    dE             = Ein[1]-Ein[0]
    E_p_bins       = np.append(E_p-0.5*(E_p[1]-E_p[0]),E_p[-1]+0.5*(E_p[1]-E_p[0]))

    dsdO_n         = RutherfordDiffXSec(Zi,Zj,mi,mj,Ei,mu)
    g_n            = 2.0/((a+b)**2-(a-b)**2)/Ei
    dsdE_p         = g_n*dsdO_n*Iin*dE

    dsdE_p,_ = np.histogram(E_ps.flatten(),bins=E_p_bins,weights=dsdE_p.flatten())

    if(normalise):
        dsdE_p /= np.trapz(dsdE_p,x=E_p)

    return E_p,dsdE_p

def mat_nuclear_elastic_knockon_ion_spectrum(mat,Ein,Iin,muc,normalise=False):

    A = mat.A

    a_2 = A/(A+1)**2
    b_2 = A/(A+1)**2
    a = np.sqrt(a_2)
    b = np.sqrt(b_2)

    Tlcoeff,Nl     = xs.interp_Tlcoeff(mat.legendre_dx_spline,Ein)
    Tlcoeff_interp = 0.5*(2*np.arange(0,Nl)+1)*Tlcoeff
    sigma          = mat.sigma(Ein)

    Ei,mu          = np.meshgrid(Ein,-muc)
    x_n            = a_2+b_2-2*a*b*mu
    E_ps           = x_n*Ei
    E_p            = (a_2+b_2+2*a*b*muc)*np.amax(Ein)
    dE             = Ein[1]-Ein[0]
    E_p_bins       = np.append(E_p-0.5*(E_p[1]-E_p[0]),E_p[-1]+0.5*(E_p[1]-E_p[0]))

    dsdO_n         = xs.diffxsec_legendre_eval(sigma,mu,Tlcoeff_interp)
    g_n            = 2.0/((a+b)**2-(a-b)**2)/Ei
    dsdE_p         = g_n*dsdO_n*Iin*dE

    dsdE_p,_ = np.histogram(E_ps.flatten(),bins=E_p_bins,weights=dsdE_p.flatten())

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