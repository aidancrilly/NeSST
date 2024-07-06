from NeSST.core import *
from NeSST.constants import *
from NeSST.utils import *

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
		print("### Init Done. ###")

	def set_primary_Tion(self,Tion):
		if(Tion < 0.1):
			print("~~ WARNING Low Tion (< 100 eV) ~~")
		self.Tion = Tion

		self.DTmean,_,self.DTvar = DTprimspecmoments(Tion)
		self.DDmean,_,self.DDvar = DDprimspecmoments(Tion)

		Y_DT = 1.0
		Y_DD = yield_from_dt_yield_ratio('dd',Y_DT,Tion)
		Y_TT = yield_from_dt_yield_ratio('tt',Y_DT,Tion)

		self.dNdE_DT = Y_DT*QBrysk(self.E_DTspec,self.DTmean,self.DTvar) # Brysk shape i.e. Gaussian
		self.dNdE_DD = Y_DD*QBrysk(self.E_sspec ,self.DDmean,self.DDvar) # Brysk shape i.e. Gaussian
		self.dNdE_TT = Y_TT*dNdE_TT(self.E_sspec,Tion)

		self.I_DT = interpolate_1d(self.E_DTspec,self.dNdE_DT,fill_value=0.0,bounds_error=False)
		self.I_DD = interpolate_1d(self.E_sspec,self.dNdE_DD)
		self.I_TT = interpolate_1d(self.E_sspec,self.dNdE_TT)

	def init_symmetric_model(self):
		"""
		Creates a callable model function for a symmetric areal density distribution
		"""

		rhoL_func = lambda x : np.ones_like(x)

		if(self.ion_kinematics):
			calc_DT_ionkin_primspec_rhoL_integral(self.dNdE_DT,nT=True,nD=True)
			mat_dict['D'].calc_n2n_dNdE(self.dNdE_DT,rhoL_func)
			mat_dict['T'].calc_n2n_dNdE(self.dNdE_DT,rhoL_func)
		else:
			mat_dict['D'].calc_station_elastic_dNdE(self.dNdE_DT,rhoL_func)
			mat_dict['T'].calc_station_elastic_dNdE(self.dNdE_DT,rhoL_func)
			mat_dict['D'].calc_n2n_dNdE(self.dNdE_DT,rhoL_func)
			mat_dict['T'].calc_n2n_dNdE(self.dNdE_DT,rhoL_func)

		dNdE_Dn2n = interpolate_1d(self.E_sspec,mat_dict['D'].n2n_dNdE,fill_value=0.0,bounds_error=False)
		dNdE_Tn2n = interpolate_1d(self.E_sspec,mat_dict['T'].n2n_dNdE,fill_value=0.0,bounds_error=False)

		if(self.ion_kinematics):
			def model(E,rhoL,vbar,dv,fT,fD,Yn):
				"""
				Symmetric areal density model with scattering ion velocity distribution with mean and std dev, vbar and dv in m/s
				"""
				A_1S = rhoR_2_A1s(rhoL,frac_D=fD,frac_T=fT)
				dNdE_nT  = mat_dict['T'].matrix_interpolate_gaussian(E,vbar,dv)
				dNdE_nD  = mat_dict['D'].matrix_interpolate_gaussian(E,vbar,dv)
				dNdE_tot = A_1S*(fT*dNdE_nT+fD*dNdE_nD+fD*dNdE_Dn2n(E)+fT*dNdE_Tn2n(E))
				return Yn*(dNdE_tot+(fD/fT)*(frac_T_default/frac_D_default)*self.I_DD(E)+(fT/fD)*(frac_D_default/frac_T_default)*self.I_TT(E))
		else:
			""" Incomplete """
			def model(E,rhoL,Ts,fT,fD,Yn):
				"""
				Symmetric areal density model with scattering temperature Ts, in keV
				"""
				A_1S = rhoR_2_A1s(rhoL,frac_D=fD,frac_T=fT)
				dNdE_nT  = mat_dict['T'].elastic_dNdE.copy()
				dNdE_nD  = mat_dict['D'].elastic_dNdE.copy()
				if(Ts > 0.1):
					T_MeV    = Ts/1e3
					E_nT0    = ((sm.A_T-1.0)/(sm.A_T+1.0))**2*self.DTmean
					dE_nT    = np.sqrt(8.0*sm.A_T*E_nT0/(sm.A_T+1.0)**2*T_MeV)
					E_nD0    = ((sm.A_D-1.0)/(sm.A_D+1.0))**2*self.DTmean
					dE_nD    = np.sqrt(8.0*sm.A_D*E_nD0/(sm.A_D+1.0)**2*T_MeV)


				dNdE_nT = interpolate_1d(self.E_sspec,dNdE_nT,fill_value=0.0,bounds_error=False)
				dNdE_nD = interpolate_1d(self.E_sspec,dNdE_nD,fill_value=0.0,bounds_error=False)

				dNdE_tot = A_1S*(fT*dNdE_nT(E)+fD*dNdE_nD(E)+fD*dNdE_Dn2n(E)+fT*dNdE_Tn2n(E))
				return Yn*(dNdE_tot+(fD/fT)*(frac_T_default/frac_D_default)*self.I_DD(E)+(fT/fD)*(frac_D_default/frac_T_default)*self.I_TT(E))

		self.model = model

	def init_modeone_model(self,P1_arr):
		"""
		Creates a callable model function for a mode 1 asymmetric areal density distribution
		"""

		self.P1_arr = P1_arr

		# T(n,2n)
		mat_dict['T'].n2n_ddx.rgrid_IE = np.trapz(mat_dict['T'].n2n_ddx.rgrid*self.dNdE_DT[:,None,None],self.E_DTspec,axis=0)
		mat_dict['T'].n2n_dNdE_mode1 = np.trapz(mat_dict['T'].n2n_ddx.rgrid_IE[:,:,None]*
		                                       (1.0+self.P1_arr[None,None,:]*mat_dict['T'].n2n_mu[:,None,None])
		                                       ,mat_dict['T'].n2n_mu,axis=0)

		mat_dict['T'].n2n_dNdE_mode1 = interpolate_2d(self.E_sspec,self.P1_arr,mat_dict['T'].n2n_dNdE_mode1,bounds_error=False)

		# D(n,2n)
		mat_dict['D'].n2n_ddx.rgrid_IE = np.trapz(mat_dict['D'].n2n_ddx.rgrid*self.dNdE_DT[:,None,None],self.E_DTspec,axis=0)
		mat_dict['D'].n2n_dNdE_mode1 = np.trapz(mat_dict['D'].n2n_ddx.rgrid_IE[:,:,None]*
		                                       (1.0+self.P1_arr[None,None,:]*mat_dict['D'].n2n_mu[:,None,None])
		                                       ,mat_dict['D'].n2n_mu,axis=0)

		mat_dict['D'].n2n_dNdE_mode1 = interpolate_2d(self.E_sspec,self.P1_arr,mat_dict['D'].n2n_dNdE_mode1,bounds_error=False)

		# nT
		M_mode1 = np.trapz((1.0+self.P1_arr[None,None,None,:]*mat_dict['T'].full_scattering_mu[:,:,:,None])
		                                *mat_dict['T'].full_scattering_M[:,:,:,None]
		                                *self.dNdE_DT[None,None,:,None],self.E_DTspec,axis=2)

		mat_dict['T'].M_mode1_interp = interpolate_1d(self.P1_arr,M_mode1,axis=-1,bounds_error=False)

		# nD
		M_mode1 = np.trapz((1.0+self.P1_arr[None,None,None,:]*mat_dict['D'].full_scattering_mu[:,:,:,None])
		                                *mat_dict['D'].full_scattering_M[:,:,:,None]
		                                *self.dNdE_DT[None,None,:,None],self.E_DTspec,axis=2)

		mat_dict['D'].M_mode1_interp = interpolate_1d(self.P1_arr,M_mode1,axis=-1,bounds_error=False)

		if(self.ion_kinematics):
			def model(E,rhoL,P1,vbar,dv,fT,fD,Yn):
				A_1S = rhoR_2_A1s(rhoL,frac_D=fD,frac_T=fT)
				mat_dict['T'].M_prim = mat_dict['T'].M_mode1_interp(P1)
				mat_dict['D'].M_prim = mat_dict['D'].M_mode1_interp(P1)
				dNdE_nT   = mat_dict['T'].matrix_interpolate_gaussian(E,vbar,dv)
				dNdE_nD   = mat_dict['D'].matrix_interpolate_gaussian(E,vbar,dv)
				dNdE_Tn2n = mat_dict['T'].n2n_dNdE_mode1(E,P1)
				dNdE_Dn2n = mat_dict['D'].n2n_dNdE_mode1(E,P1)
				dNdE_tot  = A_1S*(fT*dNdE_nT+fD*dNdE_nD+fD*dNdE_Dn2n+fT*dNdE_Tn2n)
				# Primary
				dNdE_DD   = (fD/fT)*(frac_T_default/frac_D_default)*self.I_DD(E)
				dNdE_TT   = (fT/fD)*(frac_D_default/frac_T_default)*self.I_TT(E)
				return Yn*(dNdE_tot+dNdE_DD+dNdE_TT)
		else:
			""" Incomplete """
			def model():
				return None
		self.model = model