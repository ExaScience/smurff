#!/bin/bash

if [ $blas_impl == "mkl" ]
then
    SKBUILD_CMAKE_ARGS="-DENABLE_MKL=ON"
elif [ $blas_impl == "openblas" ]
then
    SKBUILD_CMAKE_ARGS="-DENABLE_OPENBLAS=ON"
else
    echo "Unknown BLAS impl: \"$blas_impl\""
    exit -1
fi

echo "extra CMAKE_ARGS: $SKBUILD_CMAKE_ARGS"

export SKBUILD_CMAKE_ARGS

$PYTHON -m pip install . --no-deps -vv
