''' Contains some physical constants used throughout the analysis '''
import scipy.constants as sc
import numpy as np
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