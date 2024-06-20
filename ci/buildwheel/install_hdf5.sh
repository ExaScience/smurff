#!/bin/bash

set -ex

BASEDIR=$PWD

# Set the HDF5 version
HDF5_MAJOR=1
HDF5_MINOR=12
HDF5_RELEASE=2
HDF5_VERSION=${HDF5_MAJOR}.${HDF5_MINOR}.${HDF5_RELEASE}
HDF5_URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_MAJOR}.${HDF5_MINOR}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"

mkdir -p hdf5_build
cd hdf5_build

wget -q $HDF5_URL -O - | tar -xvzf - --strip-components 2

./configure --prefix=/usr/local/hdf5
make -j4
sudo make install