#!/bin/bash

export CMAKE_ARGS="-DENABLE_MKL=ON -DCMAKE_INSTALL_PREFIX=$PREFIX -DENABLE_MPI=OFF -DENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo"

$PYTHON setup.py install \
    --install-binaries \
    --single-version-externally-managed --record=record.txt