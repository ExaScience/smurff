#!/bin/bash
set -e

# Install Eigen
echo "Installing Eigen..."
wget -O Eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
tar xzf Eigen.tar.gz
rm Eigen.tar.gz
cd eigen*
mkdir build
cd build
cmake ..
make -j2
make install
cd ../..
rm -r eigen*

# Install HighFive
echo "Installing HighFive..."
wget -O HighFive.tar.gz https://github.com/BlueBrain/HighFive/archive/v2.10.1.tar.gz
tar xzf HighFive.tar.gz
rm HighFive.tar.gz
cd HighFive*
mkdir build
cd build
cmake .. -DHIGHFIVE_USE_BOOST=OFF
make -j2
make install
cd ../..
rm -r HighFive*

echo "Done!"
