from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages

import os
import sys
import subprocess

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS"
]

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', extra_cmake_args='', extra_build_args=''):
        Extension.__init__(self, name, sources=[])
        print("Created CMake Extension:", vars(self))
        self.sourcedir = os.path.abspath(sourcedir)
        self.extra_cmake_args = extra_cmake_args.split()
        self.extra_build_args = extra_build_args.split()

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'

        os.makedirs(self.build_temp, exist_ok = True)
        cmake_args = [
                        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                        '-DPYTHON_EXECUTABLE=' + sys.executable,
                        '-DCMAKE_BUILD_TYPE=' + cfg,
                        ] + ext.extra_cmake_args

        cmake_srcdir = ext.sourcedir
        subprocess.check_call(['cmake', cmake_srcdir] + cmake_args, cwd=self.build_temp)

        build_args = ['--config', cfg] + ext.extra_build_args
        subprocess.check_call(['cmake', '--build', '.' ] + build_args, cwd=self.build_temp)

        if install_binaries:
            subprocess.check_call(['cmake', '--build', '.', '--target', 'install'] + build_args, cwd=self.build_temp)

extra_cmake_args = ''
extra_build_args = ''
install_binaries = False

if "--extra-cmake-args" in sys.argv:
    index = sys.argv.index('--extra-cmake-args')
    sys.argv.pop(index)
    extra_cmake_args += " " + sys.argv.pop(index)

if "CMAKE_ARGS" in os.environ:
    extra_cmake_args += " " + os.environ.get("CMAKE_ARGS")

if "--extra-build-args" in sys.argv:
    index = sys.argv.index('--extra-build-args')
    sys.argv.pop(index)
    extra_build_args += " " + sys.argv.pop(index)

if "BUILD_ARGS" in os.environ:
    extra_build_args += " " + os.environ.get("BUILD_ARGS")

try:
    index = sys.argv.index('--install-binaries')
    sys.argv.pop(index)
    install_binaries = True
except ValueError:
    pass

setup(
    name = 'smurff',
    package_dir={'':'python/smurff'},
    packages = [ 'smurff' ],
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'dirty-tag'
    },
    url = "http://github.com/ExaScience/smurff",
    ext_modules=[CMakeExtension('smurff/wrapper', '.', extra_cmake_args, extra_build_args)],
    cmdclass=dict(build_ext=CMakeBuild), 
    zip_safe = False,
    license = "MIT",
    description = 'Bayesian Factorization Methods',
    long_description = 'Highly optimized and parallelized methods for Bayesian Factorization, including BPMF and smurff. The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices. smurff method provides also the ability to incorporate high-dimensional side information to the factorization.',
    author = "Tom Vander Aa",
    author_email = "Tom.VanderAa@imec.be",
    classifiers = CLASSIFIERS,
    keywords = "bayesian factorization machine-learning high-dimensional side-information",
    install_requires = [ 'numpy', 'scipy', 'pandas', 'scikit-learn', 'hdf5' ],
    setup_requires=['setuptools_scm', 'pybind11' ],
)

