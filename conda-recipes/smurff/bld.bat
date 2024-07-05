if "%blas_impl%"=="mkl" (
    set "SKBUILD_CMAKE_ARGS=-DBLA_VENDOR=Intel10_64_dyn"
) else (
    set "SKBUILD_CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS"
)

echo extra CMAKE_ARGS: %SKBUILD_CMAKE_ARGS%

%PYTHON% -m pip install . --no-deps -vv
if errorlevel 1 exit 1