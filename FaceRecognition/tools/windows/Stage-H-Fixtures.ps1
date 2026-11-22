param([Parameter(Mandatory=$true)][string]$Port)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$Python = Get-InstrumentPython $Workspace
$Tool = Join-Path $Workspace 'FaceRecognition/tools/stage_h_exact_fixtures.py'
& $Python $Tool '--port' $Port
if ($LASTEXITCODE -ne 0) { throw 'Exact H fixture transfer/readback failed; keep the device unchanged and save output.' }
