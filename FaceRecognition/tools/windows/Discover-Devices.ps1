. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
Invoke-StudyPython $Workspace @('-m','nicla_study.cli','discover','--workspace',$Workspace)
$Capture = Join-Path $Workspace 'FaceRecognition/nicla_study/instruments/windows_fnirsi/capture_windows.py'
$Python = Get-InstrumentPython $Workspace
& $Python $Capture '--project-root' $Workspace '--list'
if ($LASTEXITCODE -ne 0) { throw 'Fresh HID inventory failed.' }
