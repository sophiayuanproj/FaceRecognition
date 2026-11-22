. (Join-Path $PSScriptRoot '_common.ps1')
$Workspace = Get-StudyWorkspaceRoot
$python = Get-InstrumentPython $Workspace
Push-Location (Join-Path $Workspace 'FaceRecognition')
try {
    & $python -c 'import importlib.util, nicla_study.cli, nicla_study.i_matrix_campaign, nicla_study.i_lifecycle_campaign, nicla_study.j_pilot_workflow, nicla_study.freeze_selection, nicla_study.annotation_workflow, nicla_study.presentation_player; from nicla_study.instruments.windows_fnirsi import capture_windows, offline_audit; assert "tensorflow" not in __import__("sys").modules; assert "h5py" not in __import__("sys").modules; print("portable Windows workflow imports pass without TensorFlow/h5py")'
    if ($LASTEXITCODE -ne 0) { throw 'Portable instrument import isolation failed.' }
}
finally { Pop-Location }
