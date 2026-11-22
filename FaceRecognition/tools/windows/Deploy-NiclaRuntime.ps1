param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$Python = Get-InstrumentPython $Workspace
$Tool = Join-Path $Workspace 'FaceRecognition/tools/deploy_nicla_runtime.py'
& $Python $Tool @Arguments
if ($LASTEXITCODE -ne 0) { throw 'Nicla deployment/preflight failed. Preserve receipts and do not delete the prior release.' }
