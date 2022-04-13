# NeSST
Neutron Scattered Spectra Tool

import NeSST as nst

## Author:
- Aidan Crilly
E-mail: ac116@ic.ac.uk

##Installation

- Install from PyPI 

pip install NeSST

- Clone git repository 

pip install -e .

## Package Description
NeSST is a tool for producing singly scattered neutron spectra from ICF implosions. Various models for primary neutron spectra are given but the main focus of the code is on the scattered components.
Total and differential cross sections for elastic and inelastic processes are used to form a singly scattered spectrum - the effect of areal density asymmetries can be incorporated into the resultant spectra.

Example of code usage can be found in ./example directory

## Current model specifications:
- Primary spectrum models for DT, DD and TT
- Elastic and inelastic (n,2n) processes for D and T
- Relativistic corrections to elastic scattering kernels
- Scattering of all primary neutron sources
- Inclusion of areal density asymmetry effects and variable fuel fractions
- Backscatter edge shape effects from scattering ion kinematics
- Preliminary scattering 9Be support

## Future model developments:
- Offline table support for backscatter edge matrix
- Additional ablator scattering support (C, CH)

## Publications
The models used in this code are described in the following publications:

The effect of areal density asymmetries on scattered neutron spectra in ICF implosions, PoP, 2021

Neutron backscatter edge: A measure of the hydrodynamic properties of the dense DT fuel at stagnation in ICF experiments, PoP, 2020

## Acknowledgements:
Many thanks to Owen Mannion and Brian Appelbe for their help during development of NeSST