#!/usr/bin/env bash

set -e

cd python/smurff


for PYVER in "cp35-cp35m" "cp36-cp36m" "cp37-cp37m" "cp38-cp38" ; do
  PYPREFIX="/opt/python/${PYVER}"
  PYINCLUDE_DIR=$(echo ${PYPREFIX}/include/python*)
  CMAKE_ARGS="-DENABLE_BOOST=OFF -DENABLE_OPENBLAS=ON -DPYTHON_INCLUDE_DIR=${PYINCLUDE_DIR}"
  "${PYPREFIX}/bin/pip" install Cython numpy 
  "${PYPREFIX}/bin/python" setup.py --extra-cmake-args "$CMAKE_ARGS" bdist_wheel
done

find dist -name "*.whl" -exec auditwheel repair {} \;
