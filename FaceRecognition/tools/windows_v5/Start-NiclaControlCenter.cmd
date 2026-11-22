@echo off
setlocal
for %%I in ("%~dp0..\..\..") do set "WORKSPACE=%%~fI"
set "APP=%WORKSPACE%\FaceRecognition"
set "PYTHON=%APP%\.venv-nicla-windows\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Run Setup-NiclaControlCenter.cmd first.
  exit /b 2
)
set "PYTHONPATH=%APP%"
pushd "%WORKSPACE%"
"%PYTHON%" -m control_center.server --port 8780
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
