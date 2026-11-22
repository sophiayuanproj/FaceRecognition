param([Parameter(Mandatory=$true)][string]$Archive, [string]$Workspace)
. (Join-Path $PSScriptRoot '_common.ps1')
if (-not $Workspace) { $Workspace = Get-StudyWorkspaceRoot }
$Python = Get-InstrumentPython $Workspace
$Tool = Join-Path $Workspace 'FaceRecognition/tools/portable_transfer.py'
& $Python $Tool verify $Archive
if ($LASTEXITCODE -ne 0) { throw 'Transfer archive hash/path verification failed.' }
