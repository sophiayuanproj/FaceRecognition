param([switch]$Execute, [switch]$Resume, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$PythonArgs = @('-m','nicla_study.i_matrix_campaign')
if ($Resume) { $PythonArgs += '--resume' }
if ($Execute) { $PythonArgs += '--execute' } else { $PythonArgs += '--dry-run' }
$PythonArgs += $Arguments
Invoke-StudyPython $Workspace $PythonArgs
