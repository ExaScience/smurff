#!/bin/sh
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release -DHIGHFIVE_USE_BOOST=OFF ..
make install
