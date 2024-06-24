#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -eCPU_COUNT=2 -v $(git rev-parse --show-toplevel):/smurff -ti smurff2204 /smurff/ci/ubuntu2204/build_script.sh
#
# where smurff2204 is the image name


set -e
set -x

rm -rf /work
mkdir /work
cd /work

git config --global --add safe.directory /smurff/.git
git clone /smurff

cd smurff
cmake -S . -B build
cmake --build build
cmake --install build

python3 -m pip install .

smurff --bist
pytest-3 python/test
