param([Parameter(Mandatory=$true)][string]$RunDirectory, [Parameter(Mandatory=$true)][string]$OutputJson)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
if (-not [System.IO.Path]::IsPathRooted($RunDirectory)) { $RunDirectory = Join-Path $Workspace $RunDirectory }
if (-not [System.IO.Path]::IsPathRooted($OutputJson)) { $OutputJson = Join-Path $Workspace $OutputJson }
Invoke-StudyPython $Workspace @('-m','nicla_study.instruments.windows_fnirsi.offline_audit','--run-dir',$RunDirectory,'--output-json',$OutputJson)
