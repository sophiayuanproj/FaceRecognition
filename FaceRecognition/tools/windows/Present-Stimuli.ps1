param([Parameter(Mandatory=$true)][ValidateSet('presentation','development','final')][string]$Block,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
Invoke-StudyPython $Workspace (@('-m','nicla_study.presentation_player','--workspace',$Workspace,'--block',$Block) + $Arguments)
