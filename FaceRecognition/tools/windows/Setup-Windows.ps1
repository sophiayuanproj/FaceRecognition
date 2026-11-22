. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$Venv = Join-Path $Workspace '.venv-instrument'
$Requirements = Join-Path $Workspace 'FaceRecognition/requirements-instrument-windows.txt'
$Launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
if (-not $Launcher) { throw 'Install the packet-verified 64-bit CPython 3.14.9 and Python Launcher, then rerun this script.' }
$Version = & $Launcher.Source -3.14 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])'
if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne '3.14.9') { throw "Expected verified CPython 3.14.9; found '$Version'. No environment was created." }
if (-not (Test-Path -LiteralPath $Venv)) { & $Launcher.Source -3.14 -m venv $Venv; if ($LASTEXITCODE -ne 0) { throw 'Could not create project-local instrument environment.' } }
$Python = Get-InstrumentPython $Workspace
& $Python -m pip install --disable-pip-version-check -r $Requirements
if ($LASTEXITCODE -ne 0) { throw 'Instrument dependency installation failed.' }
& $Python -c 'import importlib.util, hid, serial, mpremote, numpy, PIL; assert importlib.util.find_spec("tensorflow") is None; assert importlib.util.find_spec("h5py") is None; print("instrument imports pass; TensorFlow/h5py absent")'
if ($LASTEXITCODE -ne 0) { throw 'Instrument import isolation check failed.' }
& $Python -m pip freeze | Set-Content -Encoding utf8 (Join-Path $Workspace 'work/project-finish-v4/reports/windows-instrument-freeze.txt')
Write-Host "Instrument environment ready at $Venv"
