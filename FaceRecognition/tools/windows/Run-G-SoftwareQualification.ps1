. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$Python = Get-InstrumentPython $Workspace
$Harness = Join-Path $Workspace 'FaceRecognition/nicla_study/instruments/windows_fnirsi/g_qualification/scripts/qualify_harness.py'
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
& $Python $Harness '--output-prefix' ("G-software-" + $Stamp)
if ($LASTEXITCODE -ne 0) { throw 'G seeded software qualification failed; do not treat software rows as hardware commissioning.' }
