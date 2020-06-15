#!/bin/bash

if [ -n "$OSX_ARCH" ]
then
    OPENMP_FLAGS="-DOpenMP_CXX_FLAG=-fopenmp=libiomp5 -DOpenMP_C_FLAG=-fopenmp=libiomp5"
else
    OPENMP_FLAGS=""
fi

CMAKE_ARGS="-DENABLE_MKL=ON $OPENMP_FLAGS -DCMAKE_INSTALL_PREFIX=$PREFIX -DENABLE_MPI=OFF"
BUILD_ARGS="-j$CPU_COUNT"

$PYTHON setup.py install \
    --install-binaries \
    --extra-cmake-args "$CMAKE_ARGS" \
    --extra-build-args "$BUILD_ARGS" \
    --single-version-externally-managed --record=record.txt