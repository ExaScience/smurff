Compilation of SMURFF
=====================

Note: the easiest way to install SMURFF is not to build it yourself. Install the binary
`Conda <https://conda.io>`__ package:

.. code:: bash

    conda install -c vanderaa smurff


When compiling SMURFF yourself, you have 3 options:


Compilation using `conda build`
-------------------------------

Conda build works on Linux, macOS and Windows. Execute

.. code:: bash

   conda build -c conda-forge -c vanderaa smurff

in the `conda-recipes` directory.

Compile the binary standalone binary `smurff` using CMake
---------------------------------------------------------

This will not create the `smurff` python package.

C++ Requirements
~~~~~~~~~~~~~~~~

- CMake 3.15 or later
- Eigen3 version 3.3.7 or later
- HighFive 2.9. from https://github.com/BlueBrain/HighFive/
- Boost 1.5x or newer

CMake Options
~~~~~~~~~~~~~

- Build type switches:
   - `-DCMAKE\_BUILD\_TYPE` - Debug/Release

- Algebra library: you can specify
    - `-DENABLE\_BLAS` - ON/OFF: BLAS acceleration for Eigen is enable by default.
    - `-DBLA_VENDOR` allows you to specify which BLAS implementation to use.

- Other: look in the top-level `CMakeFile.txt` for more options.

Python package using pip
------------------------

The python package is built using scikit-build-core <https://github.com/scikit-build/scikit-build-core>,
which calls CMake to compile the C++ extension. Hence you can simply run:

.. code:: bash

   pip install .


Linux and macOs Specific
------------------------

Have a look in `ci/ <../ci/>`__ for Docker build scripts and for Linux+macOS wheel scripts. These scripts should
give you a good idea on how to compiler on an Ubuntu and macOS system.

