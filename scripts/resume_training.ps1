$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

Write-Host "Starting/resuming TinyGPT 80M from the latest safe checkpoint..."
& $Python -m transformer.train_model --resume auto
if ($LASTEXITCODE -ne 0) {
    throw "Training exited with code $LASTEXITCODE. The last periodic checkpoint is retained."
}
