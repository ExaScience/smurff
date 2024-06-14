#!/bin/bash

# Check if the version parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

# Set the HDF5 version
HDF5_VERSION=$1

if [ $HDF5_VERSION == "detect" ]; then
    HDF5_VERSION=$(python -c "import h5py; print('.'.join(map(str, h5py.h5.get_libversion())))")
    echo "Detected version: $HDF5_VERSION"
fi

# Download URL
HDF5_URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION:0:1}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"

# Create a directory for the download
mkdir -p hdf5_build
cd hdf5_build

# Download the specified version of HDF5
echo "Downloading HDF5 version $HDF5_VERSION..."
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
./configure --prefix=/usr/local/hdf5-$HDF5_VERSION

echo "Building HDF5 version $HDF5_VERSION..."
make

echo "Installing HDF5 version $HDF5_VERSION..."
sudo make install

# Cleanup
cd ../..
rm -rf hdf5_build

echo "HDF5 version $HDF5_VERSION installed successfully!"
