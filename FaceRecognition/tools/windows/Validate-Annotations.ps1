param([Parameter(Mandatory=$true)][ValidateSet('template','validate')][string]$Action,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
Invoke-StudyPython $Workspace (@('-m','nicla_study.annotation_workflow',$Action) + $Arguments)
