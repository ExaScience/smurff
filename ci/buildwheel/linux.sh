#!/usr/bin/env bash

export CIBW_BUILD='cp3*-manylinux_x86_64'
export CIBW_ENVIRONMENT='CMAKE_ARGS="-DENABLE_BOOST=OFF -DENABLE_CMDLINE=OFF -DENABLE_TESTS=OFF"'
export CIBW_MANYLINUX_X86_64_IMAGE=vanderaa/manylinux2014_x86_64_smurff
export CIBW_TEST_COMMAND='python -m unittest discover {project}/python/test'

cibuildwheel --platform linux .
