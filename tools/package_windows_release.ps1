param(
    [Parameter(Mandatory=$true)][string]$GodotPath,
    [Parameter(Mandatory=$true)][string]$TemplatePath,
    [Parameter(Mandatory=$true)][string]$NativeDirectory,
    [ValidatePattern('^\d+\.\d+\.\d+(-[A-Za-z0-9.]+)?$')][string]$Version='0.1.0-preview.2',
    [switch]$AllowDirty
)
$ErrorActionPreference='Stop'
$workspace=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dirty=[bool](& git -C $workspace status --porcelain)
if($dirty -and -not $AllowDirty) {throw 'Release packaging requires a clean commit. Use -AllowDirty only for local verification, never publication.'}
$releaseRoot=Join-Path $workspace 'build/windows-release'
$stage=Join-Path $releaseRoot ('stage-'+[DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))
$project=Join-Path $stage 'project'
$output=Join-Path $stage ('TadpolePond-'+$Version)
New-Item -ItemType Directory -Force -Path $project,$output | Out-Null
$source=Join-Path $workspace 'godot-project'
foreach($file in Get-ChildItem -LiteralPath $source -File -Recurse -Force) {
    $relative=$file.FullName.Substring($source.Length+1)
    if($relative -match '(^|\\)\.godot\\|^addons\\cudarl\\bin\\') {continue}
    $target=Join-Path $project $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $target
}
$native=Join-Path $project 'addons/cudarl/bin'
New-Item -ItemType Directory -Force -Path $native | Out-Null
foreach($dll in @('cudarl.dll','cudarl_pond_cuda.dll')) {
    Copy-Item -LiteralPath (Join-Path $NativeDirectory $dll) -Destination $native
}
$config=[IO.File]::ReadAllText((Join-Path $project 'project.godot'))
$config=$config.Replace('config/name="CudaRL Arena"','config/name="CudaRL Arena"' + "`nconfig/version=""$Version""")
$config+="`n[cudarl]`nrelease_build=true`n"
[IO.File]::WriteAllText((Join-Path $project 'project.godot'),$config,[Text.UTF8Encoding]::new($false))
New-Item -ItemType Directory -Force -Path (Join-Path $output 'data'),(Join-Path $project 'qa') | Out-Null
Copy-Item -LiteralPath (Join-Path $workspace 'docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin') -Destination (Join-Path $output 'data/selected-policy.bin')
Copy-Item -LiteralPath (Join-Path $source 'pond/policies') -Destination (Join-Path $output 'data/policies') -Recurse -Force
Copy-Item -LiteralPath (Join-Path $workspace 'tests/godot/live_feeding_smoke.gd') -Destination (Join-Path $project 'qa/live_feeding_smoke.gd')
if(Test-Path -LiteralPath (Join-Path $workspace 'tests/godot/release_menu_smoke.gd')) {
    Copy-Item -LiteralPath (Join-Path $workspace 'tests/godot/release_menu_smoke.gd') -Destination (Join-Path $project 'qa/release_menu_smoke.gd')
}
$template=(Resolve-Path $TemplatePath).Path.Replace('\','/')
$exe=(Join-Path $output 'TadpolePond.exe').Replace('\','/')
@"
[preset.0]
name="Windows Desktop"
platform="Windows Desktop"
runnable=true
advanced_options=false
dedicated_server=false
custom_features=""
export_filter="all_resources"
include_filter="*.bin,*.json"
exclude_filter=""
export_path="$exe"
encryption_include_filters=""
encryption_exclude_filters=""
encrypt_pck=false
encrypt_directory=false

[preset.0.options]
custom_template/debug="$template"
custom_template/release="$template"
binary_format/embed_pck=false
binary_format/architecture="x86_64"
codesign/enable=false
application/modify_resources=false
texture_format/bptc=true
texture_format/s3tc=true
texture_format/etc2=false
texture_format/etc=false
"@ | Set-Content -LiteralPath (Join-Path $project 'export_presets.cfg') -Encoding UTF8
$log=Join-Path $releaseRoot 'export'
$process=Start-Process -FilePath $GodotPath -ArgumentList @('--headless','--path',('"'+$project+'"'),'--import','--export-release','"Windows Desktop"',('"'+$exe+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
$process.WaitForExit();$process.Refresh()
if($process.ExitCode -ne 0 -or (Get-Content "$log.err" -Raw) -match 'SCRIPT ERROR|ERROR:') {throw "Export failed; inspect $log.err"}
$licenses=Join-Path $output 'licenses'
New-Item -ItemType Directory -Force -Path $licenses | Out-Null
foreach($component in @('godot','cuda','nsis')) {Copy-Item -LiteralPath (Join-Path $workspace "docs/third-party/$component") -Destination $licenses -Recurse -Force}
@"
Tadpole Pond $Version — Windows preview

Extract the complete ZIP. Open TadpolePond.exe. Keep its PCK, data, addons and license folders together.
No Godot editor, Python, PyTorch or CUDA toolkit installation is required.

Open pond, then click the water to drop food. Pellets sink and drift. The creature chooses and swims independently while PPO practices your conditions.
CUDA is preferred when its device/kernel check succeeds. Otherwise genuine CPU PPO uses 16 practice swimmers. The panel reports the actual backend. Learning speed depends on your computer.
Pause learning leaves the visible pond running. Autosave is every 30 seconds and on graceful exit; reopen to resume. Saves stay on your computer under Godot's CudaRL Arena app-data folder. CPU and CUDA use separate checkpoints. Run one learning window at a time.

This preview is unsigned; Windows may show a publisher/reputation warning. Download only from the project's official website/GitHub release. No paid GPU server or account is used.
64-bit Windows 10/11 and current graphics drivers are required. NVIDIA is optional; CUDA acceleration in this build targets RTX 30-series or newer (Ampere or newer). Older cards use CPU learning. Physical growth/legs and macOS/browser editions are future work.

Source: https://github.com/life423/CudaRL-Arena
Third-party redistribution notices are included in licenses.
"@ | Set-Content -LiteralPath (Join-Path $output 'START-HERE.txt') -Encoding UTF8
$manifest=[ordered]@{version=$Version;sourceCommit=(& git -C $workspace rev-parse HEAD).Trim();sourceDirty=$dirty;engine='Godot 4.4.1';cpuLearning=$true;cudaArchitectures=@('80','86','89','120');files=@()}
foreach($file in Get-ChildItem -LiteralPath $output -File -Recurse) {
    $manifest.files+=@{path=$file.FullName.Substring($output.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'release-manifest.json') -Encoding UTF8
$archive=Join-Path $releaseRoot 'TadpolePond-Windows-x64.zip'
Compress-Archive -Path (Join-Path $output '*') -DestinationPath $archive -Force
$hash=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  TadpolePond-Windows-x64.zip" | Set-Content -LiteralPath (Join-Path $releaseRoot 'SHA256SUMS.txt') -Encoding ascii
[pscustomobject]@{project=$project;output=$output;archive=$archive;bytes=(Get-Item -LiteralPath $archive).Length;sha256=$hash} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $releaseRoot 'package-result.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $releaseRoot 'package-result.json')
