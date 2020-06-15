#!/bin/bash

rm -rf build 
mkdir build
cd build

if [ -n "$OSX_ARCH" ]
then
    OPENMP_FLAGS="-DOpenMP_CXX_FLAG=-fopenmp=libiomp5 -DOpenMP_C_FLAG=-fopenmp=libiomp5"
else
    OPENMP_FLAGS=""
fi


cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DENABLE_MKL=ON $OPENMP_FLAGS

make -j$CPU_COUNT
make install
cd python/smurff

$PYTHON setup.py install --single-version-externally-managed --record=record.txt
