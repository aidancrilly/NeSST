Installation
============

Requirements
------------

NeSST requires Python 3 and the following packages:

- `NumPy <https://numpy.org/>`_
- `SciPy <https://scipy.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `endf <https://github.com/paulromano/endf-python>`_

Installing from PyPI
--------------------

The easiest way to install NeSST is via pip from the Python Package Index:

.. code-block:: bash

   pip install NeSST

Installing from source
----------------------

To install from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/aidancrilly/NeSST.git
   cd NeSST
   pip install -e .

Verifying the installation
--------------------------

After installation, you can verify NeSST is installed correctly:

.. code-block:: python

   import NeSST as nst
