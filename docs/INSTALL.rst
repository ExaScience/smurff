Compilation of SMURFF
=====================

Note: the easiest way to install SMURFF is not to build it yourself. Install the binary 
`Conda <https://conda.io>`__ package:

.. code:: bash

    conda install -c vanderaa smurff

Compilation using `conda build`
-------------------------------

Conda build works on Linux, macOS and Windows. Execute

.. code:: bash

   conda build smurff
   
in the `conda-recipes` directory.

Compilation using CMake
-----------------------

C++ Requirements
~~~~~~~~~~~~~~~~

- CMake 3.6 or later
- Eigen3 version 3.3.7 or later 
- HighFive 2.2. from https://github.com/BlueBrain/HighFive/ 
- Boost 1.5x or newer

Python Requirements
~~~~~~~~~~~~~~~~~~~

As in setup.py:

    install_requires = [ 'numpy', 'scipy', 'pandas', 'scikit-learn', 'h5sparse-tensor' ],
    setup_requires = ['setuptools_scm', 'pybind11' ],

Compile using setup.py
~~~~~~~~~~~~~~~~~~~~~~

Running 
    setup.py install

will run CMake to configure, compile and install SMURFF.
Extra arguments to CMake can be passed with 
    setup.py --extra-cmake-args <...> install
    
or by setting the `CMAKE_ARGS` environment variables.

CMake Options
~~~~~~~~~~~~~

- Build type switches:
   - CMAKE\_BUILD\_TYPE - Debug/Release

- Algebra library switches (select only one):
    - When no switches are specified, CMake will try to find
      any LAPACK and BLAS library on your system.
    - ENABLE\_OPENBLAS - ON/OFF (should include openblas
      library when linking. openblas also contains
      implementation of lapack called relapack)
    - ENABLE\_MKL - ON/OFF: tries to find the `MKL single dynamic
      library <https://software.intel.com/en-us/mkl-linux-developer-guide-using-the-single-dynamic-library>`_.

- Python:
   - ENABLE\_PYTHON

Linux and macOs Specific 
~~~~~~~~~~~~~~~~~~~~~~~~

Have a look in `ci/ <../ci/>`__ for Docker build scripts and for Linux+macOS wheel scripts. These scripts should
give you a good idea on how to compiler on an Ubuntu and macOS system.

Windows Specific 
~~~~~~~~~~~~~~~~

Work for a vcpkg-based build is in progress.
