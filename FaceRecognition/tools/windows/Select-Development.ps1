param([Parameter(Mandatory=$true)][string]$InputLog, [Parameter(Mandatory=$true)][string]$OutputReport)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
Invoke-StudyPython $Workspace @('-m','nicla_study.development_selector','--input',$InputLog,'--output',$OutputReport)
