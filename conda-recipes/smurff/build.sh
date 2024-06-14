#!/bin/bash

if [ "$blas_impl" == "mkl" ]
then
    SKBUILD_CMAKE_ARGS="-DENABLE_MKL=ON"
else
    SKBUILD_CMAKE_ARGS="-DENABLE_OPENBLAS=ON"
fi

echo "extra CMAKE_ARGS: $SKBUILD_CMAKE_ARGS"

export SKBUILD_CMAKE_ARGS

$PYTHON -m pip install . --no-deps -vv
