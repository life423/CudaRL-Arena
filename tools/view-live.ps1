param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$project = Join-Path $repo 'live'
$godotCommand = Get-Command 'Godot*_win64.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $godotCommand) { throw 'Godot is not on PATH. Open live/project.godot after building frozen_swimmer.' }
if (-not $SkipBuild) {
    & cmake -S $repo -B (Join-Path $repo 'build') -DCUDARL_BUILD_LIVE_DEMO=ON
    if ($LASTEXITCODE -ne 0) { throw 'Live demo configuration failed' }
    & cmake --build (Join-Path $repo 'build') --config Release --target frozen_swimmer --parallel 2
    if ($LASTEXITCODE -ne 0) { throw 'Live demo build failed' }
}
if (-not (Test-Path -LiteralPath (Join-Path $project 'addons/frozen_swimmer/bin/frozen_swimmer.dll'))) { throw 'Build frozen_swimmer first' }
# Interactive launch only: the scene starts paused; no verification or training runs.
& $godotCommand.Source --path $project
