param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$project = Join-Path $repo 'live'
$godotCommand = Get-Command 'Godot*_win64.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $godotCommand) { throw 'Godot is not on PATH. Open live/project.godot after building frozen_swimmer.' }
if (-not $SkipBuild) {
    $timing=& (Join-Path $PSScriptRoot 'build-native.ps1') -Target frozen_swimmer
    Write-Host ('Native build: configure {0:N3}s; build {1:N3}s; explicit configure={2}' -f $timing.configure_seconds,$timing.build_seconds,$timing.explicit_configure)
}
if (-not (Test-Path -LiteralPath (Join-Path $project 'addons/frozen_swimmer/bin/frozen_swimmer.dll'))) { throw 'Build frozen_swimmer first' }
# Interactive launch only: the scene starts paused; no verification or training runs.
& $godotCommand.Source --path $project
