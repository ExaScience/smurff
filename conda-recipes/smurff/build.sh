#!/bin/bash

export CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$PREFIX -DENABLE_MPI=OFF"

echo "extra CMAKE_ARGS: $SKBUILD_CMAKE_ARGS"

$PYTHON -m pip install . --no-deps -vv