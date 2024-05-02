import numpy as np
import pandas as pd
import json
import NeSST as nst
import matplotlib.pyplot as plt

f = open(f"{nst.data_dir}TT_xsec_ENDF.json")
TT_data = pd.DataFrame(json.load(f)['datasets'][0]['pts'])

TT_data['K'] = TT_data['E']/2

# Kinematics
import scipy.constants as sc
m1 = nst.Mt
m2 = nst.Mt
Z1 = 1.0
Z2 = 1.0

m12 = m1*m2/(m1+m2)
KB = 2*m12*(np.pi*sc.fine_structure*Z1*Z2)**2

S = TT_data['Sig']*TT_data['K']*np.exp(np.sqrt(KB/TT_data['K']))
S[0] = S[1]
K = TT_data['K'].copy()
K[0] = 0.0

K_MeV = K/1e6
S_MeVb = S/1e6

# Fitting
from scipy.optimize import minimize
def Pade_MSE(theta,x,y,n):
    a = theta[:n]
    b = theta[n:]
    ymodel = Pade(a,b,x)
    return np.sum((y-ymodel)**2)

def Pade(a,b,x):
    b_dom = np.append(b,1.0)
    return np.polyval(a,x)/np.polyval(b_dom,x)

def step_up_nfit(nlow,nup,a0,x,y,tol=1e-9):
    n_fit = nlow
    a_fit = np.zeros(n_fit)
    b_fit = np.zeros(n_fit-1)
    a_fit[-1] = a0
    theta_init = np.concatenate((a_fit,b_fit))
    opt = minimize(lambda theta: Pade_MSE(theta,x,y,n_fit),theta_init,tol=tol)

    for n_fit in range(nlow+1,nup+1):
        a_fit = np.insert(opt.x[:n_fit-1],0,0.0)
        b_fit = np.insert(opt.x[n_fit-1:],0,0.0)
        theta_init = np.concatenate((a_fit,b_fit))
        opt = minimize(lambda theta: Pade_MSE(theta,x,y,n_fit),theta_init,tol=tol)

    return opt,n_fit

opt_S,n_fit1 = step_up_nfit(2,6,0.18,K_MeV,S_MeVb)

plt.figure()
plt.subplot(121)

plt.semilogx(K,S,label='ENDF')
plt.plot(K,1e6*Pade(opt_S.x[:n_fit1],opt_S.x[n_fit1:],K_MeV),'k--',label='Pade fit')

plt.ylim(0.0,0.8e6)
plt.ylabel("S (eV b)")
plt.xlabel('K (eV)')

plt.subplot(122)
plt.semilogx(K,S-1e6*Pade(opt_S.x[:n_fit1],opt_S.x[n_fit1:],K_MeV))
plt.ylabel("Residual")
plt.xlabel('K (eV)')

plt.tight_layout()

# Effective S-factor
def S_effective_Pade_MSE(theta,model,x,y,n):
    a = theta[:n]
    b = theta[n:]
    ymodel = model(x)*Pade(a,b,x)
    return np.sum((y-ymodel)**2)

Ti = np.logspace(2,5,500)

S_Pade = lambda K : 1e6*Pade(opt_S.x[:n_fit1],opt_S.x[n_fit1:],K/1e6)

Seff_arr_Pade = nst.sm.Seff_calc(Ti,S_Pade,nst.Mt,nst.Mt,1.0,1.0,x_upper=100.0)

opt_S_eff,n_fit2 = step_up_nfit(2,5,0.18,Ti/1e6,Seff_arr_Pade/1e6)

from scipy.interpolate import interp1d
S_interp = interp1d(K,S,fill_value='extrapolate')

Seff_arr_interp = nst.sm.Seff_calc(Ti,S_interp,nst.Mt,nst.Mt,1.0,1.0,x_upper=100.0)

opt_S_eff,n_fit2 = step_up_nfit(2,3,0.18,Ti/1e6,Seff_arr_interp/1e6)

plt.figure()
plt.subplot(121)

plt.semilogx(Ti,Seff_arr_Pade)
plt.semilogx(Ti,Seff_arr_interp)
plt.plot(Ti,1e6*Pade(opt_S_eff.x[:n_fit2],opt_S_eff.x[n_fit2:],Ti/1e6),'k--',label='Pade fit')
plt.ylabel("S_eff (eV b)")
plt.xlabel('T (eV)')
plt.xlim(Ti[0],Ti[-1])

plt.subplot(122)
plt.semilogx(Ti,Seff_arr_interp-1e6*Pade(opt_S_eff.x[:n_fit2],opt_S_eff.x[n_fit2:],Ti/1e6))
plt.xlim(Ti[0],Ti[-1])
plt.ylabel("Residual")
plt.xlabel('K (eV)')

plt.tight_layout()

def print_Pade(theta,n,scale):
    a = theta[:n][::-1]
    b = theta[n:][::-1]

    b = np.insert(b,0,1.0)
    print(a[::-1])
    print(b[::-1])
    print('a:')
    for i in range(n):
        print(f'{a[i]/scale**i}',end='\t')
    print('\nb:')
    for i in range(n):
        print(f'{b[i]/scale**i}',end='\t')
    print('\n')

print('S(K) fit')
print_Pade(opt_S.x,n_fit1,1.0)

print('S_eff(Ti) fit')
print_Pade(opt_S_eff.x,n_fit2,1.0)

plt.show()