SET "CMAKE_ARGS=-DENABLE_MKL=ON -DBoost_COMPILER=-v140 -DCMAKE_INSTALL_PREFIX=%PREFIX%"
SET "BUILD_ARGS=-j"

%PYTHON% setup.py install ^
    --install-binaries ^
    --single-version-externally-managed --record=record.txt