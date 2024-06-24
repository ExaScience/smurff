%CONDA_PREFIX%\Scripts\smurff --bist ~[random]
if errorlevel 1 exit 1
%PYTHON% -m pytest
if errorlevel 1 exit 1