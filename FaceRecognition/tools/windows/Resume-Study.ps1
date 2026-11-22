param([string]$Phase)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$PythonArgs = @('-m','nicla_study.cli','resume','--workspace',$Workspace)
if ($Phase) { $PythonArgs += @('--phase',$Phase) }
Invoke-StudyPython $Workspace $PythonArgs
