%CONDA_PREFIX%\Scripts\smurff --version --verbose=1
if errorlevel 1 exit 1
%CONDA_PREFIX%\Scripts\smurff --bist ~[random]
if errorlevel 1 exit 1
%PYTHON% -m pytest -n auto -v
if errorlevel 1 exit 1