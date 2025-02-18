#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

./vcpkg/bootstrap-vcpkg.sh