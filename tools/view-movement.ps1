$ErrorActionPreference = 'Stop'
$replayProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../replay'))
$godotCommand = Get-Command 'Godot*_win64.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $godotCommand) { throw 'Godot is not on PATH. Open replay/project.godot with Godot.' }
& $godotCommand.Source --path $replayProject
