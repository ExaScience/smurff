#!/bin/bash

if [ $blas_impl == "mkl" ]
then
    SKBUILD_CMAKE_ARGS="-DENABLE_MKL=ON"
elif [ $blas_impl == "openblas" ]
then
    SKBUILD_CMAKE_ARGS="-DENABLE_OPENBLAS=ON"
fi

export SKBUILD_CMAKE_ARGS

$PYTHON -m pip install . \
    $EXTRA_CMAKE_ARGS \
    --no-deps -vv
