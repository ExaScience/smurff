#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
cd "$SCRIPT_DIR"

echo "MACOSX_DEPLOYMENT_TARGET: [$MACOSX_DEPLOYMENT_TARGET]"
echo "PWD: [$PWD]"

# ./install_hdf5.sh
./install_openblas.sh