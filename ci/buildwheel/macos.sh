#!/usr/bin/env bash

set -e

brew update
for BREW_PKG in pyenv eigen lapack pybind11 ci/highfive.rb
do
    brew outdated $BREW_PKG || brew upgrade $BREW_PKG
done


export MACOSX_DEPLOYMENT_TARGET=10.9
export PATH=~/.pyenv/shims:$PATH
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/usr/local/opt/lapack -DENABLE_BOOST=OFF -DENABLE_CMDLINE=OFF -DENABLE_TESTS=OFF"

for PYVER in "3.5.9" "3.6.10" "3.7.7" "3.8.2"; do
  pyenv install --skip-existing ${PYVER}
  pyenv global ${PYVER}
  python -m pip install wheel numpy delocate
  python setup.py bdist_wheel
done

mkdir -p wheelhouse
delocate-wheel -w wheelhouse dist/*.whl

