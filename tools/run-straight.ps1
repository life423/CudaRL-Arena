param(
    [ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Name = (Get-Date -AsUTC -Format 'yyyyMMdd-HHmmss'),
    [ValidateRange(1,64)][int]$Updates = 25,
    [ValidateSet(32,80)][int]$RolloutControls = 80,
    [ValidateRange(0,1)][double]$Lambda = 1,
    [uint32]$Seed = 0x53574D32,
    [ValidateRange(2,12)][double]$SpeedMmS = 8,
    [ValidateRange(1,20)][int]$WallSeconds = 20,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
if ($Updates*16*$RolloutControls -gt 32768) { throw 'Training allocation exceeds approved control ceiling' }
$setupWatch = [Diagnostics.Stopwatch]::StartNew()
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base = Join-Path $repo 'build/straight-motor'
$output = [IO.Path]::GetFullPath((Join-Path $base $Name))
if (-not $output.StartsWith($base + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Run path must stay inside build/straight-motor' }
if (Test-Path -LiteralPath $output) { throw 'Run identity already exists; no retry or overwrite' }
New-Item -ItemType Directory -Path $base -Force | Out-Null
$latest = Join-Path $base 'latest-status.json'
function Write-Status([string]$Phase, [hashtable]$Extra = @{}) {
    $value = @{phase=$Phase;task=4;model=6;run=$output;utc=(Get-Date -AsUTC -Format o);rollout_controls=$RolloutControls;gae_lambda=$Lambda;controls_max=(32+$Updates*16*$RolloutControls+640);native_substeps_max=(10*(32+$Updates*16*$RolloutControls+640));wall_seconds_max=$WallSeconds}
    foreach ($key in $Extra.Keys) { $value[$key]=$Extra[$key] }
    $value | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $latest
}
Write-Status 'building'
if (-not $SkipBuild) {
    $cache = Join-Path $repo 'build/CMakeCache.txt'
    if (-not (Test-Path $cache) -or -not (Select-String -LiteralPath $cache -Pattern '^CUDARL_BUILD_STROKE_MOTOR:BOOL=ON$' -Quiet)) {
        & cmake -S $repo -B (Join-Path $repo 'build') -DCUDARL_BUILD_STROKE_MOTOR=ON
        if ($LASTEXITCODE -ne 0) { Write-Status 'configure_failed'; throw 'Focused target configuration failed; no run registered' }
    }
    & cmake --build (Join-Path $repo 'build') --config Release --target stroke_motor_demo --parallel 2
    if ($LASTEXITCODE -ne 0) { Write-Status 'build_failed'; throw 'Focused build failed; no run registered' }
}
$executable = Get-Item -LiteralPath (Join-Path $repo 'build/Release/stroke_motor_demo.exe')
$sources = @(Get-ChildItem (Join-Path $repo 'src') -File -Recurse) + @(Get-Item (Join-Path $repo 'tools/stroke_motor_demo.cu'))
$compiledSources = @($sources | Where-Object { $_.FullName -notmatch '[\\/]godot[\\/]' -and $_.Name -notin @('contact_events.h','contact_pilot.h','contact_response.h','frozen_controller.cpp','frozen_controller.h','frozen_contract.in.h') })
if (@($compiledSources | Where-Object LastWriteTimeUtc -gt $executable.LastWriteTimeUtc).Count) { throw 'Executable is older than relevant source; build before registering' }
New-Item -ItemType Directory -Path (Join-Path $output 'source') -Force | Out-Null
$manifest = @()
foreach ($file in ($sources + @(Get-Item $PSCommandPath,(Join-Path $repo 'CMakeLists.txt')))) {
    $relative = [IO.Path]::GetRelativePath($repo,$file.FullName)
    $target = Join-Path (Join-Path $output 'source') $relative
    New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $target
    $manifest += @{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
}
$manifest | ConvertTo-Json | Set-Content (Join-Path $output 'source-manifest.json')
$frozenExe = Join-Path $output 'stroke_motor_demo.exe'
Copy-Item -LiteralPath $executable.FullName -Destination $frozenExe
$speed = $SpeedMmS.ToString('R',[Globalization.CultureInfo]::InvariantCulture)
$arguments = @('--updates',"$Updates",'--rollout-controls',"$RolloutControls",'--lambda',$Lambda.ToString('R',[Globalization.CultureInfo]::InvariantCulture),'--seed',"$Seed",'--speed-mm-s',$speed,'--wall-seconds',"$WallSeconds")
& $frozenExe --prepare $output @arguments
if ($LASTEXITCODE -ne 0) { Write-Status 'registration_failed'; throw 'Registration failed; no physical run launched' }
Get-FileHash $frozenExe,(Join-Path $output 'protocol.bin'),(Join-Path $output 'source-manifest.json') | Select-Object Path,Hash | ConvertTo-Json | Set-Content (Join-Path $output 'frozen-hashes.json')
$setupSeconds = $setupWatch.Elapsed.TotalSeconds
Write-Status 'running' @{setup_seconds=$setupSeconds;result_path=(Join-Path $output 'result.json')}
$processWatch = [Diagnostics.Stopwatch]::StartNew()
$processArguments = @('--run',('"'+$output+'"')) + $arguments
$ownProcess = Start-Process -FilePath $frozenExe -ArgumentList $processArguments -WindowStyle Hidden -RedirectStandardOutput (Join-Path $output 'native.log') -RedirectStandardError (Join-Path $output 'native.err') -PassThru
while (-not $ownProcess.WaitForExit(500)) {
    if ($processWatch.Elapsed.TotalSeconds -gt ($WallSeconds+5)) { Stop-Process -Id $ownProcess.Id; Write-Status 'wrapper_timeout'; throw 'Own bounded process stopped; inspect partial evidence, no retry' }
    $extra = @{setup_seconds=$setupSeconds;process_seconds=$processWatch.Elapsed.TotalSeconds}
    $progress = Join-Path $output 'training.csv'
    if (Test-Path $progress) { $last=Get-Content $progress -Tail 1; if ($last -match '^([0-9]+),([0-9]+),') {$extra.completed_updates=[int]$Matches[1];$extra.completed_training_controls=[int]$Matches[2]} }
    Write-Status 'running' $extra
}
$summary = Join-Path $output 'summary.json'
if (-not (Test-Path -LiteralPath $summary)) { Write-Status 'no_summary' @{exit_code=$ownProcess.ExitCode}; throw 'Process ended without summary; do not retry blindly' }
$result = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
$result | Add-Member setup_seconds $setupSeconds
$result | Add-Member process_seconds $processWatch.Elapsed.TotalSeconds
$result | Add-Member run_path $output
$baselineCost = if ($result.before_mean_forward_m -gt 0) {$result.before_mean_work_J/$result.before_mean_forward_m} else {$null}
$candidateCost = if ($result.after_mean_forward_m -gt 0) {$result.after_mean_work_J/$result.after_mean_forward_m} else {$null}
# Frozen promotion rule: four passes, better return, no worse speed error or
# positive work per metre. A completed optimizer run does not imply promotion.
$promote = [string]::IsNullOrEmpty($result.fault) -and $result.after_successes -eq 4 -and
    $result.after_mean_return -gt ($result.before_mean_return+0.01) -and
    $result.after_speed_RMSE_m_s -le $result.before_speed_RMSE_m_s -and
    $null -ne $candidateCost -and $null -ne $baselineCost -and $candidateCost -le $baselineCost
$selection = @{selected=$(if($promote){'learned-candidate'}else{'passing-default'});initial_policy='initial-policy.bin';candidate_checkpoint="update-$Updates.bin";runtime_adopted=$false;baseline_work_J_per_m=$baselineCost;candidate_work_J_per_m=$candidateCost;promotion_rule='4/4, return improves >0.01, no worse speed RMSE or work/metre'}
$selection | ConvertTo-Json | Set-Content (Join-Path $output 'selected-controller.json')
$result | Add-Member selected_controller $selection.selected
$result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $output 'result.json')
Write-Status 'completed' @{result_path=(Join-Path $output 'result.json');setup_seconds=$setupSeconds;native_seconds=$result.wall_seconds;fault=$result.fault}
# Result first. Preservation analysis/docs/viewers are deliberately outside this command.
Write-Output ($result | ConvertTo-Json -Depth 8 -Compress)
exit $ownProcess.ExitCode
