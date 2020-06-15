#!/usr/bin/env bash

set -e

brew update
for BREW_PKG in pyenv openblas 
do
    brew outdated $BREW_PKG || brew upgrade $BREW_PKG
done


export MACOSX_DEPLOYMENT_TARGET=10.9
export PATH=~/.pyenv/shims:$PATH
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/usr/local/opt/openblas -DENABLE_OPENBLAS=ON -DENABLE_BOOST=OFF "

for PYVER in "3.5.9" "3.6.10" "3.7.7" "3.8.2"; do
  pyenv install --skip-existing ${PYVER}
  pyenv global ${PYVER}
  python -m pip install Cython wheel numpy delocate
  python setup.py --extra-cmake-args "$CMAKE_ARGS" bdist_wheel
done

mkdir -p fixed_wheels
delocate-wheel -w fixed_wheels dist/*.whl

