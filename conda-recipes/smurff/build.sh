#!/bin/bash

if [ "$blas_impl" == "mkl" ]
then
    SKBUILD_CMAKE_ARGS="-DENABLE_MKL=ON -DENABLE_OPENBLAS=OFF"
else
    SKBUILD_CMAKE_ARGS="-DENABLE_OPENBLAS=ON -DENABLE_MKL=OFF"
fi

echo "extra CMAKE_ARGS: $SKBUILD_CMAKE_ARGS"

export SKBUILD_CMAKE_ARGS

$PYTHON -m pip install . --no-deps -vv
