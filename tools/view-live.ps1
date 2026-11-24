param([switch]$SkipBuild,[string]$GodotExe=$env:CUDARL_GODOT_EXE)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$project = Join-Path $repo 'live'
if ([string]::IsNullOrWhiteSpace($GodotExe)) {
    $installedGodot=Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Godot/Godot.exe'
    if (Test-Path -LiteralPath $installedGodot -PathType Leaf) {
        $GodotExe=$installedGodot
    } else {
        $godotCommand=Get-Command 'Godot*_win64.exe','godot.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $godotCommand) { throw 'Godot4.4.1 was not found. Set CUDARL_GODOT_EXE or pass -GodotExe with its executable path.' }
        $GodotExe=$godotCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $GodotExe -PathType Leaf)) { throw 'The selected Godot executable does not exist.' }
$engineVersion=(& $GodotExe --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $engineVersion -notmatch '^4\.4\.1(\.|$)') {
    throw ('This native app requires Godot4.4.1; selected engine reports '+$engineVersion)
}
if (-not $SkipBuild) {
    $timing=& (Join-Path $PSScriptRoot 'build-native.ps1') -Target frozen_swimmer
    Write-Host ('Native build: configure {0:N3}s; build {1:N3}s; explicit configure={2}' -f $timing.configure_seconds,$timing.build_seconds,$timing.explicit_configure)
}
if (-not (Test-Path -LiteralPath (Join-Path $project 'addons/frozen_swimmer/bin/frozen_swimmer.dll'))) { throw 'Build frozen_swimmer first' }
# Interactive launch only: the scene starts paused; no verification or training runs.
& $GodotExe --path $project
