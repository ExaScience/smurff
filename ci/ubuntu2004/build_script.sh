#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -eCPU_COUNT=2 -v $(git rev-parse --show-toplevel):/smurff -ti smurff2004 /smurff/ci/ubuntu2004/build_script.sh
#
# where smurff1910 is the image name


set -e
set -x

rm -rf /work  && mkdir /work && cd /work
git clone /smurff && cd smurff
python3 setup.py install --install-binaries
/usr/local/libexec/tests
pytest-3 python/test
