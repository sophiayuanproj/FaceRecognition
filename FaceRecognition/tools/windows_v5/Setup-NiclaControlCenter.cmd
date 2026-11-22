@echo off
setlocal
for %%I in ("%~dp0..\..\..") do set "WORKSPACE=%%~fI"
set "APP=%WORKSPACE%\FaceRecognition"
set "VENV=%APP%\.venv-nicla-windows"
set "WHEELS=%WORKSPACE%\work\project-research-v5\handoffs\windows_wheels"
set "NICLA_PYTHON_COMMAND=py.exe -3.12"
if defined NICLA_PYTHON_312 set NICLA_PYTHON_COMMAND="%NICLA_PYTHON_312%"
%NICLA_PYTHON_COMMAND% -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8" >nul 2>nul
if errorlevel 1 (
  echo Install 64-bit CPython 3.12 or set NICLA_PYTHON_312 to its full python.exe path.
  exit /b 2
)
if not exist "%VENV%\Scripts\python.exe" %NICLA_PYTHON_COMMAND% -m venv "%VENV%"
if errorlevel 1 exit /b 2
if not exist "%WHEELS%\pyserial-3.5-py2.py3-none-any.whl" goto :install_online
if not exist "%WHEELS%\mpremote-1.29.0-py3-none-any.whl" goto :install_online
if not exist "%WHEELS%\pillow-12.3.0-cp312-cp312-win_amd64.whl" goto :install_online
if not exist "%WHEELS%\platformdirs-4.14.2-py3-none-any.whl" goto :install_online
echo Installing pinned packages from the bundled wheels.
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check --no-index --find-links "%WHEELS%" --require-hashes -r "%APP%\requirements-live-control-center-windows.lock"
goto :installed
:install_online
echo Installing pinned packages from the Python package index. Internet access is required.
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check --require-hashes -r "%APP%\requirements-live-control-center-windows.lock"
:installed
if errorlevel 1 (
  echo Dependency installation failed. Preserve the pip output above and retry setup.
  exit /b 2
)
set "PYTHONPATH=%APP%"
pushd "%WORKSPACE%"
"%VENV%\Scripts\python.exe" -m control_center.server --doctor --port 0
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" echo Doctor failed. Preserve this output and do not open the board.
exit /b %RESULT%
