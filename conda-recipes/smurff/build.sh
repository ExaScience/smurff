#!/bin/bash

export CMAKE_GENERATOR="Ninja"

if [ "$blas_impl" == "mkl" ]
then
    SKBUILD_CMAKE_ARGS="-DBLA_VENDOR=Intel10_64_dyn"
else
    SKBUILD_CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS"
fi

echo "extra CMAKE_ARGS: $SKBUILD_CMAKE_ARGS"

export SKBUILD_CMAKE_ARGS

$PYTHON -m pip install . --no-deps -vv
