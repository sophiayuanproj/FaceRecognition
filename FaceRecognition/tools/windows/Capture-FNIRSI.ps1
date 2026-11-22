param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$Python = Get-InstrumentPython $Workspace
$Capture = Join-Path $Workspace 'FaceRecognition/nicla_study/instruments/windows_fnirsi/capture_windows.py'
& $Python $Capture '--project-root' $Workspace @Arguments
if ($LASTEXITCODE -ne 0) { throw 'FNIRSI capture failed; preserve its immutable raw run and inspect its receipt.' }
