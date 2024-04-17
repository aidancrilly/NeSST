[![Continuous Integration](https://github.com/aidancrilly/NeSST/actions/workflows/ci.yml/badge.svg)](https://github.com/aidancrilly/NeSST/actions/workflows/ci.yml)

# NeSST
Neutron Scattered Spectra Tool: A python tool for ICF neutron spectroscopy in the single scatter regime

```
import NeSST as nst
```

A quick overview is given in the [Jupyter notebook guide](https://nbviewer.org/github/aidancrilly/NeSST/blob/master/example/NeSST%20Guide.ipynb)

## Package Description
NeSST is a tool for producing singly scattered neutron spectra from ICF implosions. Various models for primary neutron spectra are given but the main focus of the code is on the scattered components.
Total and differential cross sections for elastic and inelastic processes are used to form a singly scattered spectrum - the effect of areal density asymmetries can be incorporated into the resultant spectra.
The effect of scattering ion velocities on the scattering kinematics are included in NeSST.

Example of code usage can be found in ./example directory

## Author:
- Aidan Crilly

E-mail: ac116@ic.ac.uk

## Installation

- Easier method: Install from PyPI 

```
pip install NeSST
```

- Clone git repository and pip install local copy

```
git clone https://github.com/aidancrilly/NeSST.git
cd NeSST
pip install -e .
```

## Current model specifications:
- Primary spectrum models for DT, DD and TT
- Elastic and inelastic (n,2n) processes for D and T
- ENDF file interface using [ENDF-python](https://github.com/paulromano/endf-python)
- Relativistic corrections to elastic scattering kernels
- Scattering of all primary neutron sources
- Inclusion of areal density asymmetry effects and variable fuel fractions
- Backscatter edge shape effects from scattering ion kinematics
- Synthetic neutron time-of-flight tools

## Future model developments:
- Fitting models with ion kinematic approximations
- Pre-computed table support for backscatter edge matrix
- Knock-on ion spectra

## Publications:
The models used in this code are described in the following publications:

The effect of areal density asymmetries on scattered neutron spectra in ICF implosions, PoP, 2021

Neutron backscatter edge: A measure of the hydrodynamic properties of the dense DT fuel at stagnation in ICF experiments, PoP, 2020

## Acknowledgements:
Many thanks to Owen Mannion and Brian Appelbe for their help during development of NeSST

Social preview logo by Hotpot AI image generator.
