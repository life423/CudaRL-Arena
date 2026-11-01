param([ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Name='joint-target-upright-ppo-v1',[switch]$Run,[switch]$SkipBuild,[switch]$Compare,[uint32]$Seed=1247037233,[ValidateSet(0,0.8)][float]$Rho=0,[string]$ResumeFrom,[string]$ResumeProtocol,[switch]$VerifyResume,[ValidateRange(1,128)][int]$AdditionalUpdates=128,[switch]$TiltMixture,[ValidateSet(160,640)][int]$EpisodeControls=160)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'run-status.psm1') -Force
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base=Join-Path $repo 'build/ppo';$output=Join-Path $base $Name;$latest=Join-Path $base 'latest-status.json';$native=$null
function Set-Budgets {
 $script:controlCap=if($Compare){640}elseif($VerifyResume){5120}else{$AdditionalUpdates*32*$EpisodeControls+3744}
 $script:substepCap=$controlCap*10
 $script:nativeCap=if($Compare){5}elseif($VerifyResume){10}elseif($EpisodeControls -eq 640){30}elseif($TiltMixture -or $AdditionalUpdates -le 32){20}else{60}
 $script:processCap=if($Compare){10}elseif($VerifyResume){15}elseif($EpisodeControls -eq 640){35}elseif($TiltMixture -or $AdditionalUpdates -le 32){25}else{75}
}
Set-Budgets
$buildTarget=if($Compare){'motor_compare'}else{'ppo_motor'};$entryFile=if($Compare){'motor_compare.cpp'}else{'ppo_motor.cu'}
function Status([string]$Phase,[hashtable]$Extra=@{}){$value=@{phase=$Phase;run=$output;utc=[DateTime]::UtcNow.ToString('o');control_cap=$controlCap;substep_cap=$substepCap;native_wall_seconds=$nativeCap;process_wall_seconds=$processCap};foreach($key in $Extra.Keys){$value[$key]=$Extra[$key]};Write-RunStatus -Path $latest -Value $value}
try{
 if($Compare -and $VerifyResume){throw 'Choose comparison or resume verification'}
 if($VerifyResume -and -not $Run -and -not $ResumeFrom){throw 'Verification requires explicit ResumeFrom and ResumeProtocol when preparing'}
 if([bool]$ResumeFrom -ne [bool]$ResumeProtocol){throw 'Supply both ResumeFrom and ResumeProtocol'}
 if(($Run -or $Compare) -and $ResumeFrom){throw 'Resume source is specified only when preparing a PPO run; Run uses its frozen parent copy'}
 New-Item -ItemType Directory -Force $base|Out-Null
 if(-not $Run){
  if(Test-Path -LiteralPath $output){throw 'Run identity exists; never overwrite'}
  $watch=[Diagnostics.Stopwatch]::StartNew();Status 'building'
  $timing=if($SkipBuild){[pscustomobject]@{build_seconds=0.;configure_seconds=0.;explicit_configure=$false}}else{& (Join-Path $PSScriptRoot 'build-native.ps1') -Target $buildTarget}
  $exe=Get-Item (Join-Path $repo ('build/Release/'+$buildTarget+'.exe'))
  $sources=@(Get-ChildItem (Join-Path $repo 'src') -Recurse -File|Where-Object {$_.FullName -notmatch '[\\/]godot[\\/]' -and $_.Extension -in '.h','.cpp','.cu','.cuh'})+@(Get-Item (Join-Path $PSScriptRoot $entryFile))
  if(@($sources|Where-Object LastWriteTimeUtc -gt $exe.LastWriteTimeUtc).Count){throw 'PPO executable stale'}
  Status 'nonphysical_checks'
  if($Compare){$checkpoint=Join-Path $repo 'build/ppo/joint-target-upright-ppo-v1/ppo-update-128.bin';if((Get-FileHash -LiteralPath $checkpoint).Hash -ne 'E9CF9D776EFFEB161916F8F6E8531D9BF08CC7BCB57797B1AEDA6F6E1AC5BD7D'){throw 'Wrong frozen comparison checkpoint'};& $exe.FullName --prepare $output $checkpoint}else{
   $prepareArguments=@($(if($VerifyResume){'--prepare-verify'}else{'--prepare'}),$output,'--updates',[string]$AdditionalUpdates,'--tilt-mixture',[string][int][bool]$TiltMixture,'--episode-controls',[string]$EpisodeControls)
   if($ResumeFrom){$prepareArguments+=@('--resume-from',[IO.Path]::GetFullPath($ResumeFrom),'--resume-protocol',[IO.Path]::GetFullPath($ResumeProtocol));if($PSBoundParameters.ContainsKey('Seed')){$prepareArguments+=@('--seed',[string]$Seed)};if($PSBoundParameters.ContainsKey('Rho')){$prepareArguments+=@('--rho',[string]$Rho)}}else{$prepareArguments+=@('--seed',[string]$Seed,'--rho',[string]$Rho)}
   & $exe.FullName @prepareArguments
  }
  if($LASTEXITCODE){throw 'Nonphysical readiness failed; no trial launched'}
  $initialState=Get-Content (Join-Path $output 'readiness.json') -Raw|ConvertFrom-Json
  if(-not $Compare){$Seed=[uint32]$initialState.seed;$Rho=[float]$initialState.rho}
  Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $output 'runner.exe')
  if(-not $Compare){@{algorithm='PPO';seed=$Seed;rho=$Rho;stochastic_policy='z=mu(s)+rho*(previous_z-mu(previous_s))+sigma*sqrt(1-rho^2)*eta; reset marginal sigma; both current and previous means recomputed under candidate theta; predecessor data follows each shuffled row; complete episodes end before optimizer; no history reset at nonterminal cuts';noise_matching='stationary latent sigma=exp(-1); decoded and filtered differences reported without retuning';critic_baseline='physical-state-only40; lambda1 full terminal return minus state baseline; not exact history-conditional value';actor=@(40,32,32,4);critic=@(40,32,32,1);action='Four tanh joint-angle targets, first yaw/pitch then second yaw/pitch; no oscillator';fresh_state=$(if($initialState.resumed){'Restore PPO actor critic BOTH Adam configs moments counters and global update; fresh true episodes/history'}else{'Fresh actor critic Adam moments counters and rollout; no imported weights'});starting_update=$initialState.starting_update;starting_actor_steps=$initialState.starting_actor_steps;starting_critic_steps=$initialState.starting_critic_steps;fixed_log_std=$initialState.log_std;actor_optimizer_config=$initialState.actor_optimizer_config;critic_optimizer_config=$initialState.critic_optimizer_config;resume_semantics=$initialState.resume_semantics;target_filter=@{frequency_Hz=1;rate_limit_rad_s=12;acceleration_limit_rad_s2=200;angle_limit_rad=1};torque_limit_Nm=0.00001;reward='task6 reward4: -dt*(I_2to8s*(vector_speed_error/8mm_s)^2 +0.05*((COM_lane/2mm)^2+(COM_depth/1mm)^2+(head_up_tilt/10deg)^2)+0.002*actual_normalized_torque_squared); lane/depth/posture/effort from first control; no clamps/teacher';PPO="32 environments,${EpisodeControls} controls/episode,${AdditionalUpdates} additional updates,2 epochs,batch256; lr0.001,eps1e-5,gradclip0.5,clip0.2,KLstop0.01,logstd-1,gamma0.99,lambda1,terminalbootstrap0";tilt_mixture=[bool]$TiltMixture;training_pose_degrees=$(if($TiltMixture){@(0,3,-3)}else{@(0)});evaluation=@{short_poses=@(0,1,2,3);long_poses=@(0,3);long_seconds=60;long_arena_m=@(1.2,.8);speed_mm_s=@(7.2,8.8);RMSE_mm_s=2.5;lane_mm=2;depth_mm=1;tilt_deg=10;path_deg=5;work_J_per_m=.001;selection='short4/4 improved RMSE AND long2/2 and all4s windows; no automatic adoption'};budget=@{axis_controls=64;short_controls=1280;long_controls=2400;training_controls=($AdditionalUpdates*32*$EpisodeControls);total_controls=$controlCap;substeps=$substepCap;updates=$AdditionalUpdates;optimizer_steps_max_each=($AdditionalUpdates*2*32*$EpisodeControls/256);native_seconds=$nativeCap;process_seconds=$processCap;retries=0}}|ConvertTo-Json -Depth 7|Set-Content (Join-Path $output 'task-contract.json')
  }
  if($VerifyResume){
   $contract=Get-Content (Join-Path $output 'task-contract.json') -Raw|ConvertFrom-Json
   $contract|Add-Member verification_only $true
   $contract.PPO='One shared on-policy5120row batch, one PPO update each uninterrupted/save-reload branch; identical counter-based randomness and minibatch order; no continuation campaign'
   $contract.budget=@{training_controls=5120;total_controls=5120;substeps=51200;PPO_updates_per_branch=1;branches=2;optimizer_steps_per_network_per_branch_max=40;native_seconds=10;process_seconds=15;retries=0}
   $contract.evaluation=@{policy_evaluations=0;same_batch_optimizer_equivalence=$true;automatic_adoption=$false}
   $contract|ConvertTo-Json -Depth 8|Set-Content (Join-Path $output 'task-contract.json')
  }
  $manifest=@();$snapshotSources=$sources+@(Get-Item (Join-Path $PSScriptRoot 'run-ppo.ps1'),(Join-Path $PSScriptRoot 'build-native.ps1'),(Join-Path $PSScriptRoot 'run-status.psm1'),(Join-Path $repo 'CMakeLists.txt'))
  foreach($file in $snapshotSources){$relative=[IO.Path]::GetRelativePath($repo,$file.FullName);$target=Join-Path (Join-Path $output 'source') $relative;New-Item -ItemType Directory -Force (Split-Path $target)|Out-Null;Copy-Item -LiteralPath $file.FullName -Destination $target;$manifest+=@{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName).Hash}}
  $manifest|ConvertTo-Json|Set-Content (Join-Path $output 'source-manifest.json')
  $inputs=@('runner.exe','source-manifest.json','protocol.bin','readiness.json','task-contract.json')+$(if($Compare){@('checkpoint.bin','source-ppo-protocol.bin')}else{@('initialization.bin','initial-actor.bin','initial-critic.bin')})
  if(-not $Compare -and $initialState.resumed){$inputs+=@('parent-checkpoint.bin','parent-protocol.bin')}
  Get-FileHash -LiteralPath @($inputs|ForEach-Object {Join-Path $output $_})|Select-Object Path,Hash|ConvertTo-Json|Set-Content (Join-Path $output 'frozen-hashes.json')
  $timing|Add-Member preparation_seconds ([Math]::Max(0.,$watch.Elapsed.TotalSeconds-$timing.configure_seconds-$timing.build_seconds));$timing|ConvertTo-Json|Set-Content (Join-Path $output 'preparation-timing.json');Status 'ready' @{physical_substeps=0;optimizer_steps=0};return
 }
 $prepared=Get-Content (Join-Path $output 'readiness.json') -Raw|ConvertFrom-Json
 if(-not $Compare){$AdditionalUpdates=[int]$prepared.PPO_additional_updates_max;$TiltMixture=[bool]$prepared.tilt_mixture;$EpisodeControls=[int]$prepared.rollout_controls;Set-Budgets}
 if([bool]$prepared.diagnostic_only -ne [bool]$Compare -or [bool]$prepared.verify_resume -ne [bool]$VerifyResume){throw 'Prepared mode differs'}
 if(-not $Compare){if($PSBoundParameters.ContainsKey('Seed') -and $Seed -ne $prepared.seed){throw 'Prepared seed differs'};$Seed=[uint32]$prepared.seed;if($PSBoundParameters.ContainsKey('Rho') -and $Rho -ne [float]$prepared.rho){throw 'Prepared rho differs'};$Rho=[float]$prepared.rho}
 if(Test-Path (Join-Path $output 'run-started')){throw 'Already attempted; no retry'}
 foreach($file in (Get-Content (Join-Path $output 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $file.Path).Hash -ne $file.Hash){throw ('Frozen input changed: '+$file.Path)}}
 $started=[DateTime]::UtcNow.ToString('o');$watch=[Diagnostics.Stopwatch]::StartNew()
 $arguments=@($(if($VerifyResume){'--verify-resume'}else{'--run'}),('"'+$output+'"'));if(-not $Compare){$arguments+=@('--seed',[string]$Seed,'--rho',[string]$Rho,'--updates',[string]$AdditionalUpdates,'--tilt-mixture',[string][int][bool]$TiltMixture,'--episode-controls',[string]$EpisodeControls)}
 $native=Start-Process -FilePath (Join-Path $output 'runner.exe') -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'native.log') -RedirectStandardError (Join-Path $output 'native.err')
 Status 'running' @{pid=$native.Id;started_utc=$started};Write-Output ('STARTED '+$started+' pid='+$native.Id)
 while(-not $native.WaitForExit(100)){if($watch.Elapsed.TotalSeconds -ge $processCap){$native.Kill();throw 'Own process cap; no retry'};try{$progress=Get-Content (Join-Path $output 'status.json') -Raw|ConvertFrom-Json;Status $progress.phase @{pid=$native.Id;started_utc=$started;updates=$progress.updates;controls=$progress.controls;physical_substeps=$progress.physical_substeps;training_started_unix_seconds=$progress.training_started_unix_seconds}}catch{}}
 $result=Get-Content (Join-Path $output 'result.json') -Raw|ConvertFrom-Json
 foreach($file in (Get-Content (Join-Path $output 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $file.Path).Hash -ne $file.Hash){throw 'Frozen input changed during run'}}
 $result|Add-Member frozen_input_hashes_unchanged $true;$result|Add-Member started_utc $started;$result|Add-Member ended_utc ([DateTime]::UtcNow.ToString('o'));$result|Add-Member process_seconds $watch.Elapsed.TotalSeconds
 $result|ConvertTo-Json -Depth 6|Set-Content (Join-Path $output 'result.json');Status $(if($result.fault){'failed'}else{'done'}) @{controls=$result.controls;physical_substeps=$result.physical_substeps;updates=$result.completed_updates;candidate_accepted=$result.candidate_accepted;fault=$result.fault};$result|ConvertTo-Json -Depth 8 -Compress|Write-Output
 if($native.ExitCode){throw 'Trial stopped; no retry'}
}catch{if($null -ne $native -and -not $native.HasExited){$native.Kill()};Status 'failed' @{error=$_.Exception.Message};throw}
