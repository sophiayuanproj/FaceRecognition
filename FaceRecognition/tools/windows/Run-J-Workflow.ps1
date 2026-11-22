param([Parameter(Mandatory=$true)][ValidateSet('prepare','register-all','score','fit-calibration','stage-calibration')][string]$Action,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
Invoke-StudyPython $Workspace (@('-m','nicla_study.j_pilot_workflow',$Action) + $Arguments)
