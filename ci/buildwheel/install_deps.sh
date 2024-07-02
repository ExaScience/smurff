#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
cd "$SCRIPT_DIR"

echo "MACOSX_DEPLOYMENT_TARGET: [$MACOSX_DEPLOYMENT_TARGET]"
echo "PWD: [$PWD]"

brew install  --formulae eigen ../highfive.rb catch2
brew uninstall --ignore-dependencies hdf5@1.10

./install_hdf5.sh