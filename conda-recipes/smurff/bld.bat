setlocal

if "%blas_impl%"=="mkl" (
    set "SKBUILD_CMAKE_ARGS=-DENABLE_MKL=ON -DENABLE_OPENBLAS=OFF"
) else (
    set "SKBUILD_CMAKE_ARGS=-DENABLE_OPENBLAS=ON -DENABLE_MKL=OFF"
)

echo extra CMAKE_ARGS: %SKBUILD_CMAKE_ARGS%

%PYTHON% -m pip install . --no-deps -vv

endlocal