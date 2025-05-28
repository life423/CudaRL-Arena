# Build script for Godot project
$godotPath = "C:\Users\aiand\Godot\Godot.exe"
$projectPath = "$PSScriptRoot"

Write-Host "Building Godot project..." -ForegroundColor Green

# Check if Godot exists
if (-not (Test-Path $godotPath)) {
    Write-Warning "Godot not found at $godotPath"
    Write-Host "Please install Godot 4.2+ or update the path in this script" -ForegroundColor Yellow
    exit 1
}

# Import/update project settings
Write-Host "Importing project..." -ForegroundColor Cyan
& $godotPath --headless --path $projectPath --quit

if ($LASTEXITCODE -eq 0) {
    Write-Host "Godot build complete!" -ForegroundColor Green
} else {
    Write-Error "Godot build failed with exit code $LASTEXITCODE"
    exit 1
}
