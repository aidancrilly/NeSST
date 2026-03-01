User Guide
==========

This guide gives an overview of NeSST's main features and how to use them.

Quick Start
-----------

.. code-block:: python

   import NeSST as nst
   import numpy as np

Primary Spectra
---------------

NeSST provides models for primary neutron spectra from DT, DD and TT fusion reactions.

Spectral Moments (Ballabio Fits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean energy and variance of the primary neutron spectrum can be calculated using
Ballabio fits (`Ballabio et al. 1998, Nucl. Fusion 38 1723 <https://doi.org/10.1088/0029-5515/38/11/310>`_):

.. code-block:: python

   Tion = 3000  # ion temperature in eV (3 keV)

   # DT primary spectrum moments
   mean_DT, stddev_DT, variance_DT = nst.DTprimspecmoments(Tion)

   # DD primary spectrum moments
   mean_DD, stddev_DD, variance_DD = nst.DDprimspecmoments(Tion)

Primary Spectral Shapes
~~~~~~~~~~~~~~~~~~~~~~~

Two shapes are available for the primary spectrum:

- **Brysk (Gaussian)**: :func:`~NeSST.core.QBrysk`
- **Ballabio (modified Gaussian)**: :func:`~NeSST.core.QBallabio`

.. code-block:: python

   Ein = np.linspace(10e6, 18e6, 1000)  # energy array in eV
   mean, _, variance = nst.DTprimspecmoments(Tion)

   # Brysk (Gaussian) shape
   I_E = nst.QBrysk(Ein, mean, variance)

   # Ballabio (modified Gaussian) shape
   I_E = nst.QBallabio(Ein, mean, variance)

Additionally, the DRESS Monte Carlo code can be used to generate primary spectra without assuming a particular shape:

.. code-block:: python

   Ein = np.linspace(10e6, 18e6, 200)  # energy array in eV

   # DRESS DT primary spectrum, bin centres provided by Ein
   I_E = nst.QDRESS_DT(Ein, Tion, n_samples = int(1e6))

Yields and Reactivities
~~~~~~~~~~~~~~~~~~~~~~~

Yield ratios between reactions can be estimated from reactivities:

.. code-block:: python

   dt_yield = 1e15
   # DD yield predicted from DT yield
   dd_yield = nst.yield_from_dt_yield_ratio("dd", dt_yield, Tion)
   # TT yield predicted from DT yield
   tt_yield = nst.yield_from_dt_yield_ratio("tt", dt_yield, Tion)

Scattered Spectra
-----------------

The main feature of NeSST is calculating the singly-scattered neutron spectrum.

Setting Up Energy Grids and Scattering Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before computing scattered spectra, energy grids and scattering matrices must be initialised:

.. code-block:: python

   # Define energy grids
   Eout = np.linspace(10e6, 16e6, 500)   # outgoing energy grid (eV)
   Ein  = np.linspace(12e6, 16e6, 500)   # incoming energy grid (eV)

   # Initialise DT scattering matrices
   nst.init_DT_scatter(Eout, Ein)

Symmetric (Isotropic) Areal Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a spherically symmetric implosion:

.. code-block:: python

   I_E = nst.QBrysk(Ein, mean, variance)
   total_spec, (nD, nT, Dn2n, Tn2n) = nst.DT_sym_scatter_spec(I_E)

Asymmetric Areal Density
~~~~~~~~~~~~~~~~~~~~~~~~~~

For an asymmetric (anisotropic) areal density distribution:

.. code-block:: python

   # rhoL_func must be a function of cos(theta), normalised such that
   # the integral over the sphere equals 1
   rhoL_func = lambda x: np.ones_like(x)  # isotropic example

   total_spec, components = nst.DT_asym_scatter_spec(I_E, rhoL_func)

Ion Kinematics
~~~~~~~~~~~~~~

The effect of scattering ion velocities can be included:

.. code-block:: python

   # Ion velocity array
   varr = np.linspace(-3e5, 3e5, 51)  # velocities in m/s

   # Initialise scattering matrices with ion kinematics
   nst.init_DT_ionkin_scatter(varr, nT=True, nD=True)

Transmission
------------

The straight-line transmission of primary neutrons through the DT fuel can be calculated:

.. code-block:: python

   rhoL = 0.2    # areal density in kg/m^2
   rhoL_func = lambda x: np.ones_like(x)

   transmission = nst.DT_transmission(rhoL, Ein, rhoL_func)

Areal Density Conversion
------------------------

Helper functions are provided to convert between areal density and scattering amplitude:

.. code-block:: python

   A_1S = nst.rhoR_2_A1s(rhoL)   # areal density -> scattering amplitude
   rhoL = nst.A1s_2_rhoR(A_1S)   # scattering amplitude -> areal density

Fitting Models
--------------

The :class:`~NeSST.fitting.DT_fit_function` class provides pre-built models for fitting
experimental data:

.. code-block:: python

   E_DTspec = np.linspace(12e6, 16e6, 500)
   E_sspec  = np.linspace(10e6, 16e6, 500)

   fit = nst.DT_fit_function(E_DTspec, E_sspec)
   fit.set_primary_Tion(Tion)
   fit.init_symmetric_model()

   # fit.model is a callable for use with scipy.optimize.curve_fit etc.

Neutron Time-of-Flight (nToF)
-----------------------------

Synthetic nToF detector signals can be generated using the :class:`~NeSST.time_of_flight.nToF` class:

.. code-block:: python

   from NeSST.time_of_flight import nToF, get_unity_sensitivity, get_transit_time_tophat_IRF

   distance = 3.0          # distance to detector in meters
   sensitivity = get_unity_sensitivity()
   IRF = get_transit_time_tophat_IRF(scintillator_thickness=0.01)

   detector = nToF(distance, sensitivity, IRF)
   t, t_norm, signal = detector.get_signal(Ein, I_E)

Materials
---------

In addition to the default D and T materials, other materials can be initialised:

.. code-block:: python

   # Available by default: H, D, T, C12, Be9
   mat = nst.init_mat_scatter(Eout, Ein, "C12")
   rhoL_func = lambda x: np.ones_like(x)
   spec = nst.mat_scatter_spec(mat, I_E, rhoL_func)

Publications
------------

The models used in NeSST are described in:

- A. Crilly et al., *The effect of areal density asymmetries on scattered neutron spectra in ICF implosions*, Physics of Plasmas, 2021
- A. Crilly et al., *Neutron backscatter edge: A measure of the hydrodynamic properties of the dense DT fuel at stagnation in ICF experiments*, Physics of Plasmas, 2020
