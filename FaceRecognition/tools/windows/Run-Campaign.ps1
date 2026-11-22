param([Parameter(Mandatory=$true)][ValidateSet('JDEV','JPILOT','KFINAL')][string]$Phase,
      [switch]$Execute, [switch]$Resume, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$PythonArgs = @('-m','nicla_study.run_campaign','--phase',$Phase)
if ($Execute) { $PythonArgs += '--execute' }
if ($Resume) { $PythonArgs += '--resume' }
$PythonArgs += $Arguments
Invoke-StudyPython $Workspace $PythonArgs
