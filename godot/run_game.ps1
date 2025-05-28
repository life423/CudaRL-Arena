# Run the game
$godotPath = "C:\Users\aiand\Godot\Godot.exe"
$projectPath = "$PSScriptRoot"

# Check if Godot exists
if (-not (Test-Path $godotPath)) {
    Write-Warning "Godot not found at $godotPath"
    Write-Host "Please install Godot 4.2+ or update the path in this script" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting CudaRL-Arena..." -ForegroundColor Cyan
& $godotPath --path $projectPath
