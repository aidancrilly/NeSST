from .constants import *
from scipy.special import erf

"""

Stopping power models from section IV:
    Zimmerman, G. B. Recent Developments in Monte Carlo Techniques. United States: N. p., 1990. Web.

"""


def fdi1h(u):

    # Reference: Fukushima, T. (2015, App. Math. Comp., 259, 698-707)
    
    if(u < 1.17683303804380831e0):
        t = u*0.849738210666018375e0
        z = t*(156377.8333056294e0+t*(48177.5705898287e0+t*(5847.07218383812e0+t*(335.3978079672194e0+t*7.84411868029912e0))))/ \
            (117762.02905535089e0+t*(-19007.26938370368e0+t*(1376.2936928453140e0+t*(-54.11372698481717e0+t))))
        y = np.log(z)
    elif(u < 3.82993088157949761e0):
        t = 0.376917874490198033e0*u-0.443569407329314587e0
        y = (489.140447310410217e0+t*(5335.07269317261966e0+t*(20169.0736140442509e0+t*(35247.8115595510907e0+t*(30462.3668614714761e0+t*(12567.9032426128967e0+t*(2131.86789357398657e0+t*93.6520172085419439e0)))))))/ \
            (656.826207643060606e0+t*(4274.82831051941605e0+t*(10555.7581310151498e0+t*(12341.8742094611883e0+t*(6949.18854413197094e0+t*(1692.19650634194002e0+t*(129.221772991589751e0+t)))))))
    elif(u < 13.3854493161866553e0):
        t = 0.104651569335924949e0*u-0.400808277205416960e0
        y = (1019.84886406642351e0+t*(9440.18255003922075e0+t*(33947.6616363762463e0+t*(60256.7280980542786e0+t*(55243.0045063055787e0+t*(24769.8354802210838e0+t*(4511.77288617668292e0+t*211.432806336150141e0)))))))/ \
            (350.502070353586442e0+t*(2531.06296201234050e0+t*(6939.09850659439245e0+t*(9005.40197972396592e0+t*(5606.73612994134056e0+t*(1488.76634564005075e0+t*(121.537028889412581e0+t)))))))
    elif(u < 53.2408277860982205e0):
        t = 0.0250907164450825724e0*u-0.335850513282463787e0
        y = (11885.8779398399498e0+t*(113220.250825178799e0+t*(408524.373881197840e0+t*(695674.357483475952e0+t*(569389.917088505552e0+t*(206433.082013681440e0+t*(27307.2535671974100e0+t*824.430826794730740e0)))))))/ \
            (1634.40491220861182e0+t*(12218.1158551884025e0+t*(32911.7869957793233e0+t*(38934.6963039399331e0+t*(20038.8358438225823e0+t*(3949.48380897796954e0+t*(215.607404890995706e0+t)))))))
    elif(u < 188.411871723022843e0):
        t = 0.00739803415638806339e0*u-0.393877462475929313e0
        y = (11730.7011190435638e0+t*(99421.7455796633651e0+t*(327706.968910706902e0+t*(530425.668016563224e0+t*(438631.900516555072e0+t*(175322.855662315845e0+t*(28701.9605988813884e0+t*1258.20914464286403e0)))))))/ \
            (634.080470383026173e0+t*(4295.63159860265838e0+t*(10868.5260668911946e0+t*(12781.6871997977069e0+t*(7093.80732100760563e0+t*(1675.06417056300026e0+t*(125.750901817759662e0+t)))))))
    else:
        v = u**(-4.e0/3.e0)
        s = 1080.13412050984017e0*v
        t = 1.e0-s
        w = (1.12813495144821933e7+t*(420368.911157160874e0+t*(1689.69475714536117e0+t)))/(s*(6088.08350831295857e0+t*(221.445236759466761e0+t*0.718216708695397737e0)))
        y = np.sqrt(w)

    return y

def DegenParam(Te, ne):
    # Returns the Fermi-Dirac degeneracy parameter
    # (de Broglie wavelength)^3/(spin degeneracy)
    prefactor = (sc.h**2/(2*np.pi*sc.m_e*sc.e))**1.5 / 2
    u   = ne * prefactor * Te**(-1.5) 
    # Degeneracy parameter, << -1 in classical limit
    # Fermi-Dirac integral inverse 1/2
    eta = fdi1h(u)
    # eta is the chemical potential divided by the electron temperature
    return eta

def ChandrasekharG(x):
    # See page 38 Helander and Sigmar
    return (erf(x)-1.1283791670955*x*np.exp(-x*x))/(2*x*x)

# Stopping power for particle p on a background of Maxwellian/Fermi-Dirac electrons
def dvdt_electron_MaynardDeutsch(Zp,mp,Ep,ne,Te):
    # Zimmerman, G. B. Recent Developments in Monte Carlo Techniques. United States: N. p., 1990. Web.
    # Equation (11)

    # Zp is charge of particle
    # mp is mass of particle in amu
    # Ep is energy of particle (eV)
    # ne is electron number density (1/m^3)
    # Te is electron temperature (eV)

    # Particle velocity, v = sqrt(2E/m)
    fqvp = np.sqrt(2.0*sc.e*Ep/mp/amu_kg)

    # Classical electrons
    # Equation (19)
    fqvth2 = 2.0*(sc.e*Te/sc.m_e)
    fqvth  = np.sqrt(fqvth2)
    
    eta = DegenParam(Te, ne)
    # Degeneracy effects
    #------------------------
    if isinstance(eta, np.ndarray):
        mask = eta > -10.0
        fqvth[mask] = (sc.h / (2*np.sqrt(np.pi)*sc.m_e) ) * (4 *  ne[mask] * (1 + np.exp(-eta[mask])))**(0.33333333)
    else:
        if eta > -10.0:
            fqvth = (sc.h / (2*np.sqrt(np.pi)*sc.m_e) ) * (4 *  ne * (1 + np.exp(-eta)))**(0.33333333)
    fqvth2 = fqvth**2
    #------------------------

    fqvth2 = 2.0*(sc.e*Te/sc.m_e)
    fqvth  = np.sqrt(fqvth2)

    # Normalised y
    yv2   = fqvp**2 / fqvth2

    # Rational function in Equation (16)
    # Horner sum
    polyy = (0.321 + 0.259*yv2 * (1 + 0.273 * yv2 * (1 + 0.707 * yv2) ) ) / (1 + 0.130 * yv2 * (1 + 0.385 * yv2) )
    #- electron plasma frequency w_pe^2 = n_e * e^2 / (eps * m_e)
    # Equation (17)
    prefactor = sc.e**2/(sc.epsilon_0 * sc.m_e)
    wpe2 = prefactor * ne
    wpe  = np.sqrt(wpe2)
    #- argument Gamma_F = 2 * m_e *  v_e^2 / (hbar * w_pe) * polyy,
    qcoulArg = (2* sc.m_e / sc.hbar) * fqvth2 / wpe * polyy
    qcoul    = 0.5 * np.log(1 + qcoulArg**2)
    #------------------------
    if isinstance(qcoul, np.ndarray):
        qcoul[qcoul < 0.001] = 0.001
    else:
        if qcoul < 0.001:
            qcoul = 0.001
    #------------------------

    qcSp = Zp * Zp * 884.3538169 * qcoul
    qAD  = qcSp * ne / mp

    qfac  = ChandrasekharG(fqvp/fqvth)/(fqvth*fqvth)

    # In m/s^2
    return qAD * qfac

# Stopping power for particle p on a background of Maxwellian/Fermi-Dirac electrons
def dEdx_electron_MaynardDeutsch(Zp,mp,Ep,ne,Te):
    # Zp is charge of particle
    # mp is mass of particle in amu
    # Ep is energy of particle (eV)
    # ne is electron number density (1/m^3)
    # Te is electron temperature (eV)

    dvdt = dvdt_electron_MaynardDeutsch(Zp,mp,Ep,ne,Te)

    # In eV/m
    return dvdt / sc.e * (mp * amu_kg)