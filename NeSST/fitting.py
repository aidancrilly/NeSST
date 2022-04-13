from NeSST.core import *
from NeSST.spectral_model import *
from NeSST.constants import *

class DT_fit_function:
	"""
	A class which constructs various simple models for the full spectrum in DT for fitting data

	User must provide energy grids (and velocity grids if including ion kinematics)

	One can create a model function from the following list of approximations:

	-- Symmetric areal density
	-- Asymmetric Mode 1 areal density

	The primary spectra are assumed isotropic and with moments defined by a single temperature

	This class doesn't represent the full set of spectrum models which can be produced by NeSST! Just a common few...
	"""
	def __init__(self,E_DTspec,E_sspec,vion_arr = None):
		self.E_DTspec = E_DTspec
		self.E_sspec  = E_sspec
		print("### Initialising data on energy grids... ###")
		init_DT_scatter(E_sspec,E_DTspec)
		if(vion_arr is not None):
			self.ion_kinematics = True
			self.vion_arr = vion_arr
			print("### Initialising scattering matrices on ion velocity grid... ###")
			init_DT_ionkin_scatter(vion_arr,nT=True,nD=True)
		else:
			self.ion_kinematics = False

	def set_primary_Tion(self,Tion):
		if(Tion < 0.1):
			print("~~ WARNING Low Tion (< 100 eV) ~~")
		self.Tion = Tion

		self.DTmean,self.DTvar = DTprimspecmoments(Tion)
		self.DDmean,self.DDvar = DDprimspecmoments(Tion)

		Y_DD = yield_from_dt_yield_ratio('dd',self.Y_DT,Tion)
		Y_TT = yield_from_dt_yield_ratio('tt',self.Y_DT,Tion)

		self.dNdE_DT = Y_DT*Qb(self.E_DTspec,self.DTmean,self.DTvar) # Brysk shape i.e. Gaussian
		self.dNdE_DD = Y_DD*Qb(self.E_sspec ,self.DDmean,self.DDvar) # Brysk shape i.e. Gaussian
		self.dNdE_TT = Y_TT*dNdE_TT(self.E_sspec,Tion)

		self.I_DT = interp1d(self.E_DTspec,self.dNdE_DT,fill_value=0.0,bounds_error=False)
		self.I_DD = interp1d(self.E_sspec,self.dNdE_DD)
		self.I_TT = interp1d(self.E_sspec,self.dNdE_TT)

	def init_symmetric_model(self):
		"""
		Creates a callable model function for a symmetric areal density distribution
		"""

		rhoL_func = lambda x : np.ones_like(x)

		if(self.ion_kinematics):
			calc_DT_ionkin_primspec_rhoL_integral(self.dNdE_DT,nT=True,nD=True)
			mat_D.calc_n2n_dNdE(self.dNdE_DT,rhoL_func)
			mat_T.calc_n2n_dNdE(self.dNdE_DT,rhoL_func)
		else:
			mat_D.calc_station_elastic_dNdE(self.dNdE_DT,rhoL_func)
			mat_T.calc_station_elastic_dNdE(self.dNdE_DT,rhoL_func)
			mat_D.calc_n2n_dNdE(self.dNdE_DT,rhoL_func)
			mat_T.calc_n2n_dNdE(self.dNdE_DT,rhoL_func)

		dNdE_Dn2n = interp1d(self.E_sspec,mat_D.n2n_dNdE,fill_value=0.0,bounds_error=False)
		dNdE_Tn2n = interp1d(self.E_sspec,mat_T.n2n_dNdE,fill_value=0.0,bounds_error=False)

		if(self.ion_kinematics):
			def model(E,rhoL,vbar,dv,fT,fD,Yn):
				"""
				Symmetric areal density model with scattering ion velocity distribution with mean and std dev, vbar and dv in m/s
				"""
				A_1S = rhoR_2_A1s(rhoL,frac_D=fD,frac_T=fT)
				dNdE_nT  = mat_T.matrix_interpolate_gaussian(E,vbar,dv)
				dNdE_nD  = mat_D.matrix_interpolate_gaussian(E,vbar,dv)
				dNdE_tot =  A_1S*(fT*dNdE_nT+fD*dNdE_nD+fD*dNdE_Dn2n(E)+fT*dNdE_Tn2n(E))
				return Yn*(dNdE_tot+(fD/fT)*(frac_T_default/frac_D_default)*self.I_DD(E)+(fT/fD)*(frac_D_default/frac_T_default)*self.I_TT(E))
		else:
			def model(E,rhoL,Ts,fT,fD,Yn):
				"""
				Symmetric areal density model with scattering temperature Ts, in keV
				"""
				A_1S = rhoR_2_A1s(rhoL,frac_D=fD,frac_T=fT)
				dNdE_nT  = mat_T.elastic_dNdE.copy()
				dNdE_nD  = mat_D.elastic_dNdE.copy()
				if(Ts > 0.1):
					T_MeV    = Ts/1e3
					E_nT0    = ((A_T-1.0)/(A_T+1.0))**2*self.DTmean
					dE_nT    = np.sqrt(8.0*A_T*E_nT0/(A_T+1.0)**2*T_MeV)
					E_nD0    = ((A_D-1.0)/(A_D+1.0))**2*self.DTmean
					dE_nD    = np.sqrt(8.0*A_D*E_nD0/(A_D+1.0)**2*T_MeV)


				dNdE_nT = interp1d(self.E_sspec,dNdE_nT,fill_value=0.0,bounds_error=False)
				dNdE_nD = interp1d(self.E_sspec,dNdE_nD,fill_value=0.0,bounds_error=False)

				dNdE_tot = A_1S*(fT*dNdE_nT(E)+fD*dNdE_nD(E)+fD*dNdE_Dn2n(E)+fT*dNdE_Tn2n(E))
				return Yn*(dNdE_tot+(fD/fT)*(frac_T_default/frac_D_default)*self.I_DD(E)+(fT/fD)*(frac_D_default/frac_T_default)*self.I_TT(E))

		self.model = model

	def init_modeone_model(self,P1_arr):
		"""
		Creates a callable model function for a mode 1 asymmetric areal density distribution
		"""
		if(self.ion_kinematics):
			def model():
				return None
		else:
			def model():
				return None
		self.model = model