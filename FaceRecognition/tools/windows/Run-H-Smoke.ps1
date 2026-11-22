param([Parameter(Mandatory=$true)][string]$Port, [Parameter(Mandatory=$true)][string]$OutputDirectory,
      [Parameter(Mandatory=$true)][string]$FirmwareSha256, [Parameter(Mandatory=$true)][string]$ConfigSha256,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Policies)
. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$Python = Get-InstrumentPython $Workspace
$Tool = Join-Path $Workspace 'FaceRecognition/tools/nicla_board_smoke.py'
if (-not $Policies) { $Policies = @('H','L','S','G','A-sense','A-model','A-risk','A-cost','A-full') }
& $Python $Tool '--port' $Port '--output-dir' $OutputDirectory '--expected-firmware-sha256' $FirmwareSha256 '--expected-config-sha256' $ConfigSha256 '--policies' @Policies
if ($LASTEXITCODE -ne 0) { throw 'Nicla smoke failed; preserve partial evidence and leave H gate pending.' }
