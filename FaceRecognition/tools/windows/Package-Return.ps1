param([Parameter(Mandatory=$true)][string]$Destination)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
if ([System.IO.Path]::IsPathRooted($Destination)) { $Output = $Destination } else { $Output = Join-Path $Workspace $Destination }
$Python = Get-InstrumentPython $Workspace
$Tool = Join-Path $Workspace 'FaceRecognition/tools/package_return.py'
& $Python $Tool '--workspace' $Workspace '--destination' $Output
if ($LASTEXITCODE -ne 0) { throw 'Return archive creation failed; preserve every raw run and inspect the error.' }
