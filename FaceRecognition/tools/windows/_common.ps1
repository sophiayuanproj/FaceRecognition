Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-StudyWorkspaceRoot {
    $candidate = Get-Item -LiteralPath $PSScriptRoot
    while ($null -ne $candidate) {
        $marker = Join-Path $candidate.FullName 'work/project-finish-v4/config/EXPERIMENT_PLAN.json'
        $app = Join-Path $candidate.FullName 'FaceRecognition/nicla_study'
        if ((Test-Path -LiteralPath $marker -PathType Leaf) -and (Test-Path -LiteralPath $app -PathType Container)) {
            return $candidate.FullName
        }
        $candidate = $candidate.Parent
    }
    throw 'Could not locate the transferred workspace root by its study markers.'
}

function Get-InstrumentPython([string]$Workspace) {
    $path = Join-Path $Workspace '.venv-instrument/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Instrument environment not found at $path. Run Setup-Windows.ps1 first."
    }
    return $path
}

function Invoke-StudyPython([string]$Workspace, [string[]]$Arguments) {
    $python = Get-InstrumentPython $Workspace
    Push-Location (Join-Path $Workspace 'FaceRecognition')
    try {
        & $python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE." }
    }
    finally { Pop-Location }
}
