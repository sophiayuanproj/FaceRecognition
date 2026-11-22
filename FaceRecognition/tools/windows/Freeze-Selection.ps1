param([Parameter(Mandatory=$true)][ValidateSet('freeze','verify')][string]$Action,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
Invoke-StudyPython $Workspace (@('-m','nicla_study.freeze_selection',$Action) + $Arguments)
