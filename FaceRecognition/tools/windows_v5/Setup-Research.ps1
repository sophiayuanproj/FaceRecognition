param([Parameter(Mandatory=$true)][string]$Archive)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not (Test-Path (Join-Path $Workspace 'work/project-research-v5/READY_FOR_WINDOWS.json'))) {
    throw 'Run this script from the extracted full v5 research transfer.'
}
$Launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
if (-not $Launcher) { throw '64-bit CPython 3.12 and the Python Launcher are required.' }
$Runtime = & $Launcher.Source -3.12 -c 'import sys,struct; print("%d.%d.%d %d" % (*sys.version_info[:3], struct.calcsize("P")*8))'
if ($LASTEXITCODE -ne 0 -or $Runtime.Trim() -notmatch '^3\.12\.[0-9]+ 64$') {
    throw "Expected 64-bit CPython 3.12; found '$Runtime'. The bundled native wheels target cp312-win_amd64."
}
$Venv = Join-Path $Workspace '.venv-windows-research'
if (-not (Test-Path -LiteralPath $Venv)) {
    & $Launcher.Source -3.12 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project-local Windows research environment.' }
}
$Python = Join-Path $Venv 'Scripts\python.exe'
$Wheels = Join-Path $Workspace 'work/project-research-v5/handoffs/windows_wheels'
$Lock = Join-Path $Workspace 'FaceRecognition/requirements-instrument-windows.lock'
& $Python -m pip install --disable-pip-version-check --no-index --find-links $Wheels --require-hashes -r $Lock
if ($LASTEXITCODE -ne 0) { throw 'Offline wheel/hash installation failed. Preserve the console output and stop before device access.' }
& $Python (Join-Path $Workspace 'FaceRecognition/tools/portable_transfer.py') verify $Archive
if ($LASTEXITCODE -ne 0) { throw 'Outbound transfer verification failed.' }
& $Python (Join-Path $Workspace 'FaceRecognition/tools/research_v5/windows_research.py') --workspace $Workspace doctor
if ($LASTEXITCODE -ne 0) { throw 'Research doctor failed. No board was opened.' }
Write-Host "Windows research environment ready: $Venv"
