SET "CMAKE_ARGS=-DENABLE_MKL=ON -DCMAKE_INSTALL_PREFIX=%PREFIX%"

%PYTHON% setup.py install ^
    --install-binaries ^
    --single-version-externally-managed --record=record.txt