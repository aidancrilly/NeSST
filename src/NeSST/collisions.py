import numpy as np
from NeSST.constants import *

classical_collisions = None

###########################
# Relativistic Collisions #
###########################

def gamma(beta):
	return 1.0/np.sqrt(1-beta**2)

def p(m,beta):
	g = gamma(beta)
	return g*m*beta

def E(m,beta):
	mom = p(m,beta)
	return np.sqrt(m**2+mom**2)

def mom_invariant(m1,m2,beta1,beta2,cos):
	E1 = E(m1,beta1)
	E2 = E(m2,beta2)
	p1 = p(m1,beta1)
	p2 = p(m2,beta2)
	return np.sqrt(m1**2+m2**2+2*E1*E2-2*p1*p2*cos)

def rel_lab_scattering_cosine(m1,m2,beta1,beta2,beta3,cos12,cos23):
	# m3 == m1
	E1 = E(m1,beta1); E2 = E(m2,beta2); E3 = E(m1,beta3)
	p1 = p(m1,beta1); p2 = p(m2,beta2); p3 = p(m1,beta3)
	W  = mom_invariant(m1,m2,beta1,beta2,cos12)
	mu0 = (E3*(E1+E2)-0.5*W**2-p2*p3*cos23+0.5*(m2**2-m1**2))/(p1*p3)
	return mu0

# With mu_in == +1, mu_out == mu_0
def rel_mu_out(m1,m2,beta1,beta2,beta3):
	# m3 == m1
	E1 = E(m1,beta1); E2 = E(m2,beta2); E3 = E(m1,beta3)
	p1 = p(m1,beta1); p2 = p(m2,beta2); p3 = p(m1,beta3)
	W  = mom_invariant(m1,m2,beta1,beta2,1.0)
	mu = (E3*(E1+E2)-0.5*W**2+0.5*(m2**2-m1**2))/(p1*p3+p2*p3)
	return mu

def rel_CoM_scattering_cosine(m1,m2,beta1,beta2,beta3,cos12,cos23):
	# m3 == m1
	# Lab frame quantities
	E1 = E(m1,beta1); E2 = E(m2,beta2); E3 = E(m1,beta3)
	p1 = p(m1,beta1); p2 = p(m2,beta2); p3 = p(m1,beta3)
	W  = mom_invariant(m1,m2,beta1,beta2,cos12)
	mu0 = (E3*(E1+E2)-0.5*W**2-p2*p3*cos23+0.5*(m2**2-m1**2))/(p1*p3)
	# CoM quantities
	betac   = np.sqrt(p1**2+p2**2+2*p1*p2*cos12)/(E1+E2)
	gammac  = gamma(betac)
	beta_p1 = (p1**2+p2*p1*cos12)/(E1+E2)
	beta_p3 = (p1*p3*mu0+p2*p3*cos23)/(E1+E2)
	# Evaluate
	numerator   = p1*p3*mu0+gammac**2*(beta_p1*beta_p3+betac**2*E1*E3-(E3*beta_p1+E1*beta_p3))
	denominator = 0.25*(W+(m1**2-m2**2)/W)**2-m1**2
	muc = numerator/denominator
	return muc

def rel_mucE_jacobian(m1,m2,beta1,beta2,beta3,cos12,cos23):
	# m3 == m1
	# Lab frame quantities
	E1 = E(m1,beta1); E2 = E(m2,beta2); E3 = E(m1,beta3)
	p1 = p(m1,beta1); p2 = p(m2,beta2); p3 = p(m1,beta3)
	W  = mom_invariant(m1,m2,beta1,beta2,cos12)
	# Evaluate
	numerator   = E2-p2*E3*cos23/p3
	denominator = 0.25*(W+(m1**2-m2**2)/W)**2-m1**2
	g = numerator/denominator
	return g

# Conversions
def Ekin_2_beta(Ek,m):
	x = Ek/m+1
	beta = gamma_2_beta(x)
	return beta

def gamma_2_beta(g):
	return np.sqrt(1.0-1.0/g**2)

def v_2_beta(v):
	return v/c

def beta_2_normtime(beta):
	return 1.0/beta

def beta_2_Ekin(beta,m):
	Etot = E(m,beta)
	return Etot-m

def Jacobian_dEdnorm_t(E,m):
	beta = Ekin_2_beta(E,m)
	gam  = gamma(beta)
	return m*(gam*beta)**3

def velocity_addition_to_Ekin(Ek,m,u):
	beta_frame = u/c
	beta = Ekin_2_beta(Ek,m)
	beta = (beta+beta_frame)/(1+beta*beta_frame)
	Ek = beta_2_Ekin(beta,m)
	return Ek

########################
# Classical Collisions #
########################

def cla_lab_scattering_cosine(A,Ein,Eout,muin,muout,vf):
	vout  = sqrtE_2_v*np.sqrt(Eout)
	vin   = sqrtE_2_v*np.sqrt(Ein)
	mu0_star = 0.5*((A+1)*np.sqrt(Eout/Ein)-(A-1)*np.sqrt(Ein/Eout))
	return mu0_star+A*vf/vout*muin-A*vf/vin*muout

# With mu_in == +1, mu_out == mu_0
def cla_mu_out(A,Ein,Eout,vf):
	vout  = sqrtE_2_v*np.sqrt(Eout)
	vin   = sqrtE_2_v*np.sqrt(Ein)
	mu0_star = 0.5*((A+1)*np.sqrt(Eout/Ein)-(A-1)*np.sqrt(Ein/Eout))
	return (mu0_star+A*vf/vout)/(1+A*vf/vin)

def cla_CoM_scattering_cosine(A,Ein,Eout,muin,muout,vf):
	vout  = sqrtE_2_v*np.sqrt(Eout)
	vin   = sqrtE_2_v*np.sqrt(Ein)
	v_ratio = (vout**2-2*vf*vout*muout+vf**2)/(vin**2-2*vf*vin*muin+vf**2)
	return (A+1)**2*v_ratio/(2*A)-(A**2+1)/(2*A)

def cla_mucE_jacobian(A,Ein,Eout,muin,muout,vf):
	vout  = sqrtE_2_v*np.sqrt(Eout)
	vin   = sqrtE_2_v*np.sqrt(Ein)
	alpha = ((A-1)/(A+1))**2
	g0    = 2.0/((1-alpha)*Ein)
	vcorr = ((1-vf*muout/vout)/(1-2*vf*muin/vin+(vf/vin)**2))
	return g0*vcorr

#######################
# Interface functions #
#######################

# Change in flux due to different relative velocity between target and scatterer
# |vn-vf|/vn
def flux_change(Ein,muin,vf):
	vin   = sqrtE_2_v*np.sqrt(Ein)
	del_f = np.sqrt(1-2*vf*muin/vin+(vf/vin)**2)
	return del_f

# Slowing down kernel
def g(A,Ein,Eout,muin,muout,vf):
	if (classical_collisions == True):
		ans   = cla_mucE_jacobian(A,Ein,Eout,muin,muout,vf)
	else:
		beta1 = Ekin_2_beta(Ein,Mn)
		beta2 = v_2_beta(vf)
		beta3 = Ekin_2_beta(Eout,Mn)
		ans   = rel_mucE_jacobian(Mn,A*Mn,beta1,beta2,beta3,muin,muout)
	return ans

# Centre of mass cosine
def muc(A,Ein,Eout,muin,muout,vf):
	if (classical_collisions == True):
		ans   = cla_CoM_scattering_cosine(A,Ein,Eout,muin,muout,vf)
	else:
		beta1 = Ekin_2_beta(Ein,Mn)
		beta2 = v_2_beta(vf)
		beta3 = Ekin_2_beta(Eout,Mn)
		ans   = rel_CoM_scattering_cosine(Mn,A*Mn,beta1,beta2,beta3,muin,muout)
	return ans

# Outgoing neutron cosine
def mu_out(A,Ein,Eout,vf):
	if (classical_collisions == True):
		ans   = cla_mu_out(A,Ein,Eout,vf)
	else:
		beta1 = Ekin_2_beta(Ein,Mn)
		beta2 = v_2_beta(vf)
		beta3 = Ekin_2_beta(Eout,Mn)
		ans   = rel_mu_out(Mn,A*Mn,beta1,beta2,beta3)
	return ans

