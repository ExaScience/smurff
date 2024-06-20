#!/bin/bash

set -ex

VERSION=0.3.27
BASEDIR=$PWD

# Define OpenBLAS GitHub repository URL
REPO_URL="https://github.com/xianyi/OpenBLAS"

# Create a directory for the build process
BUILD_DIR="openblas_build_$VERSION"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Download the specified version of OpenBLAS
echo "Downloading OpenBLAS version $VERSION..."
wget -qO- "$REPO_URL/archive/refs/tags/v$VERSION.tar.gz" | tar xz --strip-components=1

# Build OpenBLAS
echo "Building OpenBLAS version $VERSION..."
make MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}

make install PREFIX=$BASEDIR/openblas

echo "OpenBLAS version $VERSION has been installed in $BASEDIR/openblas"