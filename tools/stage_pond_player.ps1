param(
    [Parameter(Mandatory=$true)][string]$GodotPath,
    [string]$Destination = 'build/portable/package',
    [switch]$CpuOnly
)
$ErrorActionPreference='Stop'
$workspace=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$package=[IO.Path]::GetFullPath((Join-Path $workspace $Destination))
$buildRoot=[IO.Path]::GetFullPath((Join-Path $workspace 'build'))
if(-not $package.StartsWith($buildRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {throw 'Package destination must be a subfolder of this repository build directory'}
$project=Join-Path $package 'player'
$native=Join-Path $project 'addons/cudarl/bin'
New-Item -ItemType Directory -Force -Path $native | Out-Null
$source=Join-Path $workspace 'godot-project'
$portableDll=Join-Path $workspace 'build/portable/player/addons/cudarl/bin/cudarl.dll'
if(-not (Test-Path -LiteralPath $portableDll)) {throw 'Run cmake --build --preset portable first'}
Copy-Item -LiteralPath $portableDll -Destination (Join-Path $native 'cudarl.dll')
$cuda=Join-Path $source 'addons/cudarl/bin/cudarl_pond_cuda.dll'
if(-not $CpuOnly -and (Test-Path -LiteralPath $cuda)) {Copy-Item -LiteralPath $cuda -Destination (Join-Path $native 'cudarl_pond_cuda.dll')}
elseif(Test-Path -LiteralPath (Join-Path $native 'cudarl_pond_cuda.dll')) {Remove-Item -LiteralPath (Join-Path $native 'cudarl_pond_cuda.dll')}
Copy-Item -LiteralPath (Join-Path $source 'addons/cudarl/cudarl.gdextension') -Destination (Join-Path $project 'addons/cudarl/cudarl.gdextension')
foreach($folder in @('pond','launcher')) {New-Item -ItemType Directory -Force -Path (Join-Path $project $folder) | Out-Null}
foreach($file in @('deep_pond.gd','deep_pond.tscn','deep_water.gd','deep_creature.gd','natural_shore.gd','pond_curve.gd','pond_geometry.gd')) {
    Copy-Item -LiteralPath (Join-Path $source "pond/$file") -Destination (Join-Path $project "pond/$file")
    if(Test-Path -LiteralPath (Join-Path $source "pond/$file.uid")) {Copy-Item -LiteralPath (Join-Path $source "pond/$file.uid") -Destination (Join-Path $project "pond/$file.uid")}
}
Get-ChildItem -LiteralPath (Join-Path $source 'pond') -File | Where-Object {$_.Name -match '\.gdshader(inc)?(\.uid)?$'} | Copy-Item -Destination (Join-Path $project 'pond')
Copy-Item -LiteralPath (Join-Path $source 'pond/policies') -Destination (Join-Path $project 'pond') -Recurse -Force
foreach($file in @('launcher.gd','launcher.tscn','demo_launch.gd','thumbnail.gd')) {Copy-Item -LiteralPath (Join-Path $source "launcher/$file") -Destination (Join-Path $project "launcher/$file")}
Copy-Item -LiteralPath (Join-Path $source 'launcher/assets') -Destination (Join-Path $project 'launcher') -Recurse -Force
Copy-Item -LiteralPath (Join-Path $source 'icon.svg') -Destination $project
$config=[IO.File]::ReadAllText((Join-Path $source 'project.godot'))
$config=$config.Replace('run/main_scene="res://launcher/launcher.tscn"','run/main_scene="res://pond/deep_pond.tscn"')
$config+="`n[cudarl]`nportable_player=true`n"
[IO.File]::WriteAllText((Join-Path $project 'project.godot'),$config,(New-Object Text.UTF8Encoding($false)))
Copy-Item -LiteralPath (Resolve-Path $GodotPath).Path -Destination (Join-Path $package 'engine.exe')
Copy-Item -LiteralPath (Join-Path $workspace 'build/portable/Release/CudaRLPond.exe') -Destination $package
$licenseSource=Join-Path $workspace 'docs/third-party/godot'
if(-not (Test-Path -LiteralPath (Join-Path $licenseSource 'LICENSE.txt'))) {throw 'Godot redistribution notices are missing'}
$licenseTarget=Join-Path $package 'licenses/godot'
New-Item -ItemType Directory -Force -Path $licenseTarget | Out-Null
Get-ChildItem -LiteralPath $licenseSource -File | Copy-Item -Destination $licenseTarget -Force
if(Test-Path -LiteralPath (Join-Path $native 'cudarl_pond_cuda.dll')) {
    $cudaLicense=Join-Path $workspace 'docs/third-party/cuda/LICENSE.txt'
    if(-not (Test-Path -LiteralPath $cudaLicense)) {throw 'The optional CUDA module requires its redistribution notice'}
    $cudaNotices=Join-Path $package 'licenses/cuda'
    New-Item -ItemType Directory -Force -Path $cudaNotices | Out-Null
    Copy-Item -LiteralPath $cudaLicense -Destination $cudaNotices -Force
}
@'
CudaRL Pond — Windows preview

Extract the complete ZIP, then open CudaRLPond.exe. No Godot, CUDA toolkit or Python installation is needed.
AI and physics prefer the optional CUDA module when its device-and-kernel check passes. Otherwise they run on the CPU. Graphics still use the available graphics hardware. The sidebar reports the selected compute backend.
This first portable build plays the same saved tadpole AI and shared physics. Live CPU training, other demos, growth, macOS and browser releases are separate follow-ups.
Optional troubleshooting: CudaRLPond.exe --pond-backend=cpu
Godot 4.4.1 engine and third-party license notices are in licenses. The engine binary is bundled for this preview.
'@ | Set-Content -LiteralPath (Join-Path $package 'START-HERE.txt') -Encoding UTF8
Write-Output "Staged $package (CPU only: $CpuOnly)"
