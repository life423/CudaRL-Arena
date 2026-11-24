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
# The ordinary player launch uses the exact verified DLL shipped in Git.
# -SkipBuild remains accepted for older shortcuts. Native development uses
# build-native.ps1 explicitly, which stages output outside the live project.
& (Join-Path $PSScriptRoot 'test-live-package.ps1') -ProjectPath $project
# Interactive launch only: the scene starts paused; no verification or training runs.
& $GodotExe --path $project
