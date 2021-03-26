# NeSST
Neutron Scattered Spectra Tool
## Author:
- Aidan Crilly

NeSST is a tool for producing singly scattered neutron spectra from ICF implosions. Various models for primary neutron spectra are given but the main focus of the code is on the scattered components.
Total and differential cross sections for elastic and inelastic processes are used to form a singly scattered spectrum - the effect of areal density asymmetries can be incorporated into the resultant spectra.

## Current model specifications:
- Primary spectrum models for DT, DD and TT
- Elastic and inelastic (n,2n) processes for D and T
- Relativistic corrections to elastic scattering kernels
- Scattering of all primary neutron sources
- Inclusion of mode 1 asymmetry effect
- Backscatter edge shape effects from scattering ion kinematics

## Future model developments:
- More general areal density treatment
- Offline table support for backscatter edge matrix

The models used in this code are described in the following publications:

The effect of areal density asymmetries on scattered neutron spectra in ICF implosions, PoP, 2021

Neutron backscatter edge: A measure of the hydrodynamic properties of the dense DT fuel at stagnation in ICF experiments, PoP, 2020

## Acknowledgements:
Many thanks to Owen Mannion and Brian Appelbe for their help during development of NeSST