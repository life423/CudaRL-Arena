param([ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Name='fresh-forward-ppo-v1',[switch]$Run,[switch]$SkipBuild)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'run-status.psm1') -Force
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base=Join-Path $repo 'build/ppo'
$output=Join-Path $base $Name
$latest=Join-Path $base 'latest-status.json'
$native=$null
function Status([string]$Phase,[hashtable]$Extra=@{}){
    $value=@{phase=$Phase;run=$output;utc=[DateTime]::UtcNow.ToString('o');control_cap=164512;substep_cap=1645120;native_wall_seconds=60;process_wall_seconds=75}
    foreach($key in $Extra.Keys){$value[$key]=$Extra[$key]}
    Write-RunStatus -Path $latest -Value $value
}
try {
    New-Item -ItemType Directory -Force $base|Out-Null
    if(-not $Run){
        if(Test-Path -LiteralPath $output){throw 'Run identity already exists; never overwrite results'}
        $watch=[Diagnostics.Stopwatch]::StartNew();Status 'building'
        $timing=if($SkipBuild){[pscustomobject]@{build_seconds=0.;configure_seconds=0.;explicit_configure=$false}}else{& (Join-Path $PSScriptRoot 'build-native.ps1') -Target ppo_motor}
        $exe=Get-Item (Join-Path $repo 'build/Release/ppo_motor.exe')
        $sources=@(Get-ChildItem (Join-Path $repo 'src') -Recurse -File|Where-Object {$_.FullName -notmatch '[\\/]godot[\\/]' -and $_.Extension -in '.h','.cpp','.cu','.cuh'})+@(Get-Item (Join-Path $PSScriptRoot 'ppo_motor.cu'))
        if(@($sources|Where-Object LastWriteTimeUtc -gt $exe.LastWriteTimeUtc).Count){throw 'PPO executable is stale; build first'}
        Status 'nonphysical_checks'
        & $exe.FullName --prepare $output
        if($LASTEXITCODE){throw 'Fresh initialization/CPU-CUDA numerical checks failed; no trial launched'}
        Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $output 'runner.exe')
        @{algorithm='PPO';training_reward='Shared task5/reward2: -0.05*(squared relative velocity error +0.05*squared lane error) in2-4s; torque effort0.002 charged throughout4s';credit_assignment='gamma0.99,lambda1; complete80-control finite episodes, terminal bootstrap0';fresh_seed=1347440433;actor_initialization='Orthogonal fresh random weights, output gain0.01, nominal amplitude0.35rad/cadence1Hz bias; no imported policy';critic_initialization='Fresh orthogonal random weights';optimizer='Fresh Adam lr0.001 eps0.00001 global gradient clipping0.5; moments and steps zero';exploration='Fixed diagonal Gaussian latent log_std=-2; exact sampled latent/logp retained';PPO='clip0.2,entropy0,value coefficient0.5,two epochs,epoch KL stop0.01';evaluation=@{poses=@(0,1,2,3);target_mm_s=8;minimum_steady_mm_s=7.2;maximum_steady_mm_s=8.8;RMSE_limit_mm_s=2.5;lateral_limit_mm=2;depth_limit_mm=1;path_limit_deg=5;tilt_limit_deg=10;work_limit_J_per_m=0.001;success='All4 frozen cases and lower mean RMSE than fresh initial policy';runtime_adoption=$false};budget=@{parity_controls=32;initial_and_final_evaluation_controls=640;training_controls=163840;controls=164512;substeps=1645120;PPO_updates=64;actor_Adam_steps_max=1280;critic_Adam_steps_max=1280;native_seconds=60;process_seconds=75;retries=0}}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $output 'task-contract.json')
        $manifest=@()
        $snapshotSources=$sources+@(Get-Item (Join-Path $PSScriptRoot 'run-ppo.ps1'),(Join-Path $PSScriptRoot 'build-native.ps1'),(Join-Path $PSScriptRoot 'run-status.psm1'),(Join-Path $repo 'CMakeLists.txt'))
        foreach($file in $snapshotSources){$relative=[IO.Path]::GetRelativePath($repo,$file.FullName);$target=Join-Path (Join-Path $output 'source') $relative;New-Item -ItemType Directory -Force (Split-Path $target)|Out-Null;Copy-Item -LiteralPath $file.FullName -Destination $target;$manifest+=@{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName).Hash}}
        $manifest|ConvertTo-Json|Set-Content (Join-Path $output 'source-manifest.json')
        Get-FileHash (Join-Path $output 'runner.exe'),(Join-Path $output 'protocol.bin'),(Join-Path $output 'initialization.bin'),(Join-Path $output 'readiness.json'),(Join-Path $output 'task-contract.json'),(Join-Path $output 'source-manifest.json')|Select-Object Path,Hash|ConvertTo-Json|Set-Content (Join-Path $output 'frozen-hashes.json')
        $timing|Add-Member preparation_seconds ([Math]::Max(0.,$watch.Elapsed.TotalSeconds-$timing.configure_seconds-$timing.build_seconds));$timing|ConvertTo-Json|Set-Content (Join-Path $output 'preparation-timing.json')
        Status 'ready' @{physical_substeps=0;optimizer_steps=0;preparation_seconds=$watch.Elapsed.TotalSeconds}
        return
    }
    if(Test-Path (Join-Path $output 'run-started')){throw 'PPO trial already attempted; no retry'}
    foreach($file in (Get-Content (Join-Path $output 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $file.Path).Hash -ne $file.Hash){throw ('Frozen input changed: '+$file.Path)}}
    $started=[DateTime]::UtcNow.ToString('o');$watch=[Diagnostics.Stopwatch]::StartNew()
    $native=Start-Process -FilePath (Join-Path $output 'runner.exe') -ArgumentList @('--run',('"'+$output+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'native.log') -RedirectStandardError (Join-Path $output 'native.err')
    Status 'running' @{pid=$native.Id;started_utc=$started};Write-Output ('STARTED '+$started+' pid='+$native.Id)
    while(-not $native.WaitForExit(100)){
        if($watch.Elapsed.TotalSeconds -ge 75){$native.Kill();throw 'Own PPO process75s cap reached; no retry'}
        try{$progress=Get-Content (Join-Path $output 'status.json') -Raw|ConvertFrom-Json;Status $progress.phase @{pid=$native.Id;started_utc=$started;updates=$progress.updates;controls=$progress.controls;physical_substeps=$progress.physical_substeps}}catch{}
    }
    $result=Get-Content (Join-Path $output 'result.json') -Raw|ConvertFrom-Json
    $result|Add-Member started_utc $started;$result|Add-Member ended_utc ([DateTime]::UtcNow.ToString('o'));$result|Add-Member process_seconds $watch.Elapsed.TotalSeconds
    $result|ConvertTo-Json -Depth 5|Set-Content (Join-Path $output 'result.json')
    Status $(if($result.fault){'failed'}else{'done'}) @{controls=$result.controls;physical_substeps=$result.physical_substeps;updates=$result.completed_updates;candidate_accepted=$result.candidate_accepted;fault=$result.fault}
    $result|ConvertTo-Json -Compress|Write-Output
    if($native.ExitCode){throw 'PPO trial stopped; evidence preserved, no retry'}
} catch {
    if($null -ne $native -and -not $native.HasExited){$native.Kill()}
    Status 'failed' @{error=$_.Exception.Message}
    throw
}
