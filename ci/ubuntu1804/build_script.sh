#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -eCPU_COUNT=2 -v $(git rev-parse --show-toplevel):/smurff -ti smurff1804 /smurff/ci/ubuntu1804/build_script.sh
#
# where smurff1804 is the image name


set -e
set -x

rm -rf /build  && mkdir /build && cd /build
cmake /smurff -DENABLE_PYTHON=OFF
make -j${CPU_COUNT}
./bin/tests