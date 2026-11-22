@echo off
setlocal
pushd "%~dp0"
set "APP=%CD%\FaceRecognition"
set "POWER=%CD%\work\project-research-v5\bench\live-pipeline-20261110\conditions\power"
set "WHEELS=%CD%\work\project-research-v5\handoffs\windows_wheels"
set "CAL_WHEELS=%CD%\work\project-research-v5\handoffs\windows_calibration_wheels"
set "PYTHONPATH=%APP%"
set "NICLA_PYTHON_COMMAND=py.exe -3.12"
if defined NICLA_PYTHON_312 set NICLA_PYTHON_COMMAND="%NICLA_PYTHON_312%"

call "%APP%\tools\windows_v5\Setup-CurrentMeasurement.cmd"
if errorlevel 1 goto :failed
if not exist ".venv-video-review\Scripts\python.exe" %NICLA_PYTHON_COMMAND% -m venv .venv-video-review
if errorlevel 1 goto :failed
if exist "%CAL_WHEELS%\opencv_python_headless-4.10.0.84-cp37-abi3-win_amd64.whl" (
  ".venv-video-review\Scripts\python.exe" -m pip install --disable-pip-version-check --no-index --find-links "%CAL_WHEELS%" --find-links "%WHEELS%" --require-hashes -r "%APP%\requirements-calibration-windows.lock"
) else (
  ".venv-video-review\Scripts\python.exe" -m pip install --disable-pip-version-check --require-hashes -r "%APP%\requirements-calibration-windows.lock"
)
if errorlevel 1 goto :failed
".venv-video-review\Scripts\python.exe" -c "import cv2, numpy, PIL; print('Calibration imports:', cv2.__version__, numpy.__version__, PIL.__version__)"
if errorlevel 1 goto :failed

"%APP%\.venv-nicla-windows\Scripts\python.exe" "%POWER%\test_analyze_video_energy.py" -v
if errorlevel 1 goto :failed
"%APP%\.venv-nicla-windows\Scripts\python.exe" -m unittest nicla_study.tests.test_hid_only_nicla_capture -v
if errorlevel 1 goto :failed
"%APP%\.venv-nicla-windows\Scripts\python.exe" -m unittest discover -s "%APP%\host\tests" -p "test_nicla_sight*.py"
if errorlevel 1 goto :failed
echo Offline bench checks passed. Open WINDOWS_VIDEO_POWER_BENCH.md for wiring and trial commands.
popd
exit /b 0

:failed
echo Offline bench setup failed. Keep this error output.
popd
exit /b 2
