param(
    [Parameter(Position = 0)]
    [string]$VideoPath = "$(Join-Path $PSScriptRoot 'video.mov')"
)

$repoRoot = $PSScriptRoot
$resourcesDir = Join-Path $repoRoot 'resources'
$targetName = '19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4'
$targetPath = Join-Path $resourcesDir $targetName
$backupPath = Join-Path $resourcesDir ($targetName + '.bak')

if (-not (Test-Path $VideoPath)) {
    throw "Target video '$VideoPath' was not found."
}

if (-not (Test-Path $targetPath)) {
    throw "Expected sample video '$targetPath' is missing; cannot proceed safely."
}

$pythonPath = 'python'
try {
    Get-Command $pythonPath -CommandType Application -ErrorAction Stop | Out-Null
}
catch {
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        $pythonPath = $venvPython
    }
    else {
        throw "Could not locate a Python executable. Ensure your Conda environment is active or install dependencies."
    }
}

Write-Host "Backing up existing demo clip to '$backupPath' ..."
Copy-Item -LiteralPath $targetPath -Destination $backupPath -Force

Write-Host "Swapping in '$VideoPath' for visualisation ..."
Copy-Item -LiteralPath $VideoPath -Destination $targetPath -Force

try {
    Write-Host "Launching visualiser ..."
    $process = Start-Process -FilePath $pythonPath `
        -ArgumentList '-m', 'sorawm.watermark_detector' `
        -WorkingDirectory $repoRoot `
        -Wait `
        -PassThru

    if ($process.ExitCode -ne 0) {
        Write-Warning "Visualiser exited with code $($process.ExitCode). Check logs for details."
    }
}
finally {
    Write-Host "Restoring original demo clip ..."
    if (Test-Path $backupPath) {
        Move-Item -LiteralPath $backupPath -Destination $targetPath -Force
    }
    else {
        Write-Warning "Backup clip '$backupPath' was not found; original demo clip was not restored."
    }
}

Write-Host "Done. Annotated output (if saved) is under 'outputs/sora_watermark_yolo_detected.mp4'."
