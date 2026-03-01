NeSST Documentation
===================

**NeSST** (Neutron Scattered Spectra Tool) is a Python package for ICF neutron spectroscopy in the single scatter regime.

.. image:: https://github.com/aidancrilly/NeSST/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/aidancrilly/NeSST/actions/workflows/ci.yml
   :alt: Continuous Integration

NeSST is a tool for producing singly scattered neutron spectra from ICF implosions. Various models
for primary neutron spectra are provided, with the main focus on the scattered components.
Total and differential cross sections for elastic and inelastic processes are used to form a singly
scattered spectrum — the effect of areal density asymmetries can be incorporated into the resultant
spectra. The effect of scattering ion velocities on the scattering kinematics are also included.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   guide
   api
   example_notebook

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
