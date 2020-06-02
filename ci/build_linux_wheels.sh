#!/usr/bin/env bash

set -e

cd python/smurff

for PYVER in "cp35-cp35m" "cp36-cp36m" "cp37-cp37m" "cp38-cp38" ; do
  PYBIN="/opt/python/${PYVER}/bin"
  "${PYBIN}/pip" install Cython
  "${PYBIN}/python" setup.py bdist_wheel
done
find dist -name "*.whl" -exec auditwheel repair {} \;
