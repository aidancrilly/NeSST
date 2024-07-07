import scipy.constants as sc
import numpy as np
import numpy.typing as npt

''' Contains some physical constants used throughout the analysis '''
# Scipy constants uses CODATA2018 database
# Need to swap from MeV to eV
amu  = sc.value('atomic mass constant energy equivalent in MeV')*1e6
c    = sc.c
Me   = sc.value('electron mass energy equivalent in MeV')*1e6
Mn   = sc.value('neutron mass energy equivalent in MeV')*1e6
Mp   = sc.value('proton mass energy equivalent in MeV')*1e6
Md   = sc.value('deuteron mass energy equivalent in MeV')*1e6
Mt   = sc.value('triton mass energy equivalent in MeV')*1e6
MHe3 = sc.value('helion mass energy equivalent in MeV')*1e6
MHe4 = sc.value('alpha particle mass energy equivalent in MeV')*1e6
MC   = 12.011*amu
MBe  = 9.012182*amu
hbar = sc.hbar
qe   = sc.e
fine_structure =  sc.fine_structure
sqrtE_2_v = np.sqrt(2*sc.e/sc.m_n)
Mn_kg = sc.m_n
sigmabarn = 1e-28 # barns to m^2
E0_DT = ((Md+Mt)**2+Mn**2-MHe4**2)/(2*(Md+Mt))-Mn
E0_DD = ((Md+Md)**2+Mn**2-MHe3**2)/(2*(Md+Md))-Mn

# Directories
import os
package_directory = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(package_directory,"./data/")
ENDF_dir = os.path.join(data_dir,"./ENDF/")

# Materials
default_mat_list = ['H','D','T','C12','Be9']
available_materials = []
mat_dict = {}