import numpy as np
import NeSST as nst

Ein  = np.linspace(13.25,14.75,100)
Eout = np.linspace(1.0,15.0,400)
vvec = np.linspace(-1200.0e3,1200.0e3,80)
P1   = np.linspace(-1.0,1.0,20)

# nT_table = nst.dsigdE_table("../dsigdE_tables/initial_test/","nT")
# nT_table.matrix_create_and_save(Ein,Eout,vvec,P1)