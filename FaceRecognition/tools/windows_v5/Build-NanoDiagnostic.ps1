param(
    [string]$Port,
    [switch]$Upload
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Sketch = Join-Path $Workspace 'work/project-research-v5/instrument/nano_meter_diagnostic'
$Output = Join-Path $Workspace 'work/project-research-v5/windows/build/nano-meter-diagnostic'
$MetadataDirectory = Join-Path $Workspace 'work/project-research-v5/windows/firmware'
$Fqbn = 'arduino:mbed:nano33ble'
$Cli = Get-Command 'arduino-cli.exe' -ErrorAction SilentlyContinue
if (-not $Cli) { $Cli = Get-Command 'arduino-cli' -ErrorAction SilentlyContinue }
if (-not $Cli) { throw 'Arduino CLI is not installed. Stop and report this Windows host prerequisite; this script does not install global tools.' }
if ($Upload -and -not $Port) { throw '-Upload requires the freshly identified Nano COM port in -Port.' }

$Details = & $Cli.Source board details --fqbn $Fqbn 2>&1
if ($LASTEXITCODE -ne 0) { throw "The installed Arduino core does not resolve $Fqbn. Do not guess another target. $Details" }
New-Item -ItemType Directory -Force -Path (Split-Path $Output -Parent), $MetadataDirectory | Out-Null
& $Cli.Source compile --fqbn $Fqbn --export-binaries --output-dir $Output $Sketch
if ($LASTEXITCODE -ne 0) { throw 'Nano diagnostic compile failed. Preserve the compiler log.' }
$Candidates = @(Get-ChildItem -LiteralPath $Output -File | Where-Object { $_.Extension -in @('.bin', '.hex') })
if ($Candidates.Count -ne 1) { throw "Expected one upload artifact in $Output; found $($Candidates.Count). Do not choose among ambiguous outputs." }
$Binary = $Candidates[0]
$Source = Join-Path $Sketch 'nano_meter_diagnostic.ino'
$SourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.ToLowerInvariant()
$BinaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Binary.FullName).Hash.ToLowerInvariant()
$CoreList = (& $Cli.Source core list | Out-String).Trim()
$BuildRecord = [ordered]@{
    schema = 'v5-nano-diagnostic-build-1'
    status = 'compiled_not_uploaded'
    fqbn = $Fqbn
    cli_version = ((& $Cli.Source version | Out-String).Trim())
    core_list = $CoreList
    source_relative_path = 'work/project-research-v5/instrument/nano_meter_diagnostic/nano_meter_diagnostic.ino'
    source_sha256 = $SourceHash
    binary_path = $Binary.FullName
    binary_sha256 = $BinaryHash
    upload_requested = [bool]$Upload
    upload_port = $(if ($Upload) { $Port } else { $null })
}
if ($Upload) {
    & $Cli.Source upload --fqbn $Fqbn --port $Port --input-dir $Output $Sketch
    if ($LASTEXITCODE -ne 0) { throw 'Upload failed. Preserve build logs; do not repeat against an unverified port.' }
    $BuildRecord.status = 'compiled_and_uploaded_to_user_selected_port'
}
$Receipt = Join-Path $MetadataDirectory 'nano-diagnostic-build.json'
$BuildRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Receipt -Encoding utf8
Write-Host "Source SHA-256: $SourceHash"
Write-Host "Binary SHA-256: $BinaryHash"
Write-Host "Binary path: $($Binary.FullName)"
Write-Host "Build receipt: $Receipt"
