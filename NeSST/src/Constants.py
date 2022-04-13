''' Contains some physical constants used throughout the analysis '''
import scipy.constants as sc
# Scipy constants uses CODATA2018 database

amu  = sc.value('atomic mass constant energy equivalent in MeV')
c    = sc.c
Me   = sc.value('electron mass energy equivalent in MeV')
Mn   = sc.value('neutron mass energy equivalent in MeV')
Mp   = sc.value('proton mass energy equivalent in MeV')
Md   = sc.value('deuteron mass energy equivalent in MeV')
Mt   = sc.value('triton mass energy equivalent in MeV')
MHe3 = sc.value('helion mass energy equivalent in MeV')
MHe4 = sc.value('alpha particle mass energy equivalent in MeV')
MC   = 12.011*amu
MBe  = 9.012182*amu
hbar = sc.hbar
qe   = sc.e
fine_structure =  sc.fine_structure