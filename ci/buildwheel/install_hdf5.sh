#!/bin/bash

set -ex

# Check if the version parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

# Set the HDF5 version
BASEDIR=$PWD

HDF5_MAJOR=1
HDF5_MINOR=12
HDF5_RELEASE=2
HDF5_VERSION=${HDF5_MAJOR}.${HDF5_MINOR}.${HDF5_RELEASE}

# Create a directory for the download
mkdir -p hdf5_build
cd hdf5_build

# Download the specified version of HDF5
echo "Downloading HDF5 version $HDF5_VERSION..."
HDF5_URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_MAJOR}.${HDF5_MINOR}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
wget $HDF5_URL -O hdf5-$HDF5_VERSION.tar.gz

if [ $? -ne 0 ]; then
  echo "Failed to download HDF5 version $HDF5_VERSION"
  exit 1
fi

# Extract the tarball
echo "Extracting HDF5 version $HDF5_VERSION..."
tar -xzvf hdf5-$HDF5_VERSION.tar.gz

cd hdf5-$HDF5_VERSION

# Configure, make, and install
echo "Configuring HDF5 version $HDF5_VERSION..."
./configure --prefix=$BASEDIR/hdf5

echo "Building HDF5 version $HDF5_VERSION..."
make -j

echo "Installing HDF5 version $HDF5_VERSION..."
make install

# Cleanup
cd ../..
rm -rf hdf5_build

echo "HDF5 version $HDF5_VERSION installed in $BASEDIR/hdf5!"
