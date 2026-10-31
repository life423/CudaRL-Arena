param([ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Name='joint-target-ppo-v1',[switch]$Run,[switch]$SkipBuild)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'run-status.psm1') -Force
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base=Join-Path $repo 'build/ppo';$output=Join-Path $base $Name;$latest=Join-Path $base 'latest-status.json';$native=$null
$controlCap=659104;$substepCap=6591040;$nativeCap=60;$processCap=75
function Status([string]$Phase,[hashtable]$Extra=@{}){$value=@{phase=$Phase;run=$output;utc=[DateTime]::UtcNow.ToString('o');control_cap=$controlCap;substep_cap=$substepCap;native_wall_seconds=$nativeCap;process_wall_seconds=$processCap};foreach($key in $Extra.Keys){$value[$key]=$Extra[$key]};Write-RunStatus -Path $latest -Value $value}
try{
 New-Item -ItemType Directory -Force $base|Out-Null
 if(-not $Run){
  if(Test-Path -LiteralPath $output){throw 'Run identity exists; never overwrite'}
  $watch=[Diagnostics.Stopwatch]::StartNew();Status 'building'
  $timing=if($SkipBuild){[pscustomobject]@{build_seconds=0.;configure_seconds=0.;explicit_configure=$false}}else{& (Join-Path $PSScriptRoot 'build-native.ps1') -Target ppo_motor}
  $exe=Get-Item (Join-Path $repo 'build/Release/ppo_motor.exe')
  $sources=@(Get-ChildItem (Join-Path $repo 'src') -Recurse -File|Where-Object {$_.FullName -notmatch '[\\/]godot[\\/]' -and $_.Extension -in '.h','.cpp','.cu','.cuh'})+@(Get-Item (Join-Path $PSScriptRoot 'ppo_motor.cu'))
  if(@($sources|Where-Object LastWriteTimeUtc -gt $exe.LastWriteTimeUtc).Count){throw 'PPO executable stale'}
  Status 'nonphysical_checks'; & $exe.FullName --prepare $output
  if($LASTEXITCODE){throw 'Nonphysical readiness failed; no trial launched'}
  Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $output 'runner.exe')
  @{algorithm='PPO';seed=1247037233;actor=@(40,32,32,4);critic=@(40,32,32,1);action='Four tanh joint-angle targets, first yaw/pitch then second yaw/pitch; no oscillator';fresh_state='actor critic Adam moments counters rollout; no imported weights';target_filter=@{frequency_Hz=1;rate_limit_rad_s=12;acceleration_limit_rad_s2=200;angle_limit_rad=1};torque_limit_Nm=0.00001;reward='task6 reward3: effort all intervals; squared relative vector speed error plus0.05*(lane/2mm)^2 plus0.05*(depth/1mm)^2 in2-8s; no teacher';PPO='32 environments,160 controls/episode,128 updates,2 epochs,batch256; lr0.001,eps1e-5,gradclip0.5,clip0.2,KLstop0.01,logstd-1,gamma0.99,lambda1,terminalbootstrap0';evaluation=@{short_poses=@(0,1,2,3);long_poses=@(0,3);long_seconds=60;long_arena_m=@(1.2,.8);speed_mm_s=@(7.2,8.8);RMSE_mm_s=2.5;lane_mm=2;depth_mm=1;tilt_deg=10;path_deg=5;work_J_per_m=.001;selection='short4/4 improved RMSE AND long2/2 and all4s windows; no automatic adoption'};budget=@{axis_controls=64;short_controls=1280;long_controls=2400;training_controls=655360;total_controls=$controlCap;substeps=$substepCap;updates=128;optimizer_steps_max_each=5120;native_seconds=60;process_seconds=75;retries=0}}|ConvertTo-Json -Depth 7|Set-Content (Join-Path $output 'task-contract.json')
  $manifest=@();$snapshotSources=$sources+@(Get-Item (Join-Path $PSScriptRoot 'run-ppo.ps1'),(Join-Path $PSScriptRoot 'build-native.ps1'),(Join-Path $PSScriptRoot 'run-status.psm1'),(Join-Path $repo 'CMakeLists.txt'))
  foreach($file in $snapshotSources){$relative=[IO.Path]::GetRelativePath($repo,$file.FullName);$target=Join-Path (Join-Path $output 'source') $relative;New-Item -ItemType Directory -Force (Split-Path $target)|Out-Null;Copy-Item -LiteralPath $file.FullName -Destination $target;$manifest+=@{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName).Hash}}
  $manifest|ConvertTo-Json|Set-Content (Join-Path $output 'source-manifest.json')
  Get-FileHash -LiteralPath @('runner.exe','source-manifest.json','protocol.bin','initialization.bin','readiness.json','task-contract.json'|ForEach-Object {Join-Path $output $_})|Select-Object Path,Hash|ConvertTo-Json|Set-Content (Join-Path $output 'frozen-hashes.json')
  $timing|Add-Member preparation_seconds ([Math]::Max(0.,$watch.Elapsed.TotalSeconds-$timing.configure_seconds-$timing.build_seconds));$timing|ConvertTo-Json|Set-Content (Join-Path $output 'preparation-timing.json');Status 'ready' @{physical_substeps=0;optimizer_steps=0};return
 }
 if(Test-Path (Join-Path $output 'run-started')){throw 'Already attempted; no retry'}
 foreach($file in (Get-Content (Join-Path $output 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $file.Path).Hash -ne $file.Hash){throw ('Frozen input changed: '+$file.Path)}}
 $started=[DateTime]::UtcNow.ToString('o');$watch=[Diagnostics.Stopwatch]::StartNew()
 $native=Start-Process -FilePath (Join-Path $output 'runner.exe') -ArgumentList @('--run',('"'+$output+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'native.log') -RedirectStandardError (Join-Path $output 'native.err')
 Status 'running' @{pid=$native.Id;started_utc=$started};Write-Output ('STARTED '+$started+' pid='+$native.Id)
 while(-not $native.WaitForExit(100)){if($watch.Elapsed.TotalSeconds -ge $processCap){$native.Kill();throw 'Own process cap; no retry'};try{$progress=Get-Content (Join-Path $output 'status.json') -Raw|ConvertFrom-Json;Status $progress.phase @{pid=$native.Id;started_utc=$started;updates=$progress.updates;controls=$progress.controls;physical_substeps=$progress.physical_substeps}}catch{}}
 $result=Get-Content (Join-Path $output 'result.json') -Raw|ConvertFrom-Json
 foreach($file in (Get-Content (Join-Path $output 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $file.Path).Hash -ne $file.Hash){throw 'Frozen input changed during run'}}
 $result|Add-Member frozen_input_hashes_unchanged $true;$result|Add-Member started_utc $started;$result|Add-Member ended_utc ([DateTime]::UtcNow.ToString('o'));$result|Add-Member process_seconds $watch.Elapsed.TotalSeconds
 $result|ConvertTo-Json -Depth 6|Set-Content (Join-Path $output 'result.json');Status $(if($result.fault){'failed'}else{'done'}) @{controls=$result.controls;physical_substeps=$result.physical_substeps;updates=$result.completed_updates;candidate_accepted=$result.candidate_accepted;fault=$result.fault};$result|ConvertTo-Json -Compress|Write-Output
 if($native.ExitCode){throw 'Trial stopped; no retry'}
}catch{if($null -ne $native -and -not $native.HasExited){$native.Kill()};Status 'failed' @{error=$_.Exception.Message};throw}
