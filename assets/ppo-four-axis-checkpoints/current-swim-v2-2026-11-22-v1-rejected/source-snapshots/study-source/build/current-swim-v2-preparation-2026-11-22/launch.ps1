param([Parameter(Mandatory=$true)][ValidateSet('Gate','Baseline','Profile','Calibrate','Train')][string]$Phase)
$ErrorActionPreference='Stop'
$studyPrep=$PSScriptRoot
$studyRoot=[IO.Path]::GetFullPath((Join-Path $studyPrep '../..'))
$studyRun=Join-Path $studyRoot 'build/ppo/current-swim-v2-2026-11-22-v1'
$studyReceipt=Join-Path $studyRun 'process-execution.json'
$studyEvents=@();$studySpent=0.0
if(Test-Path -LiteralPath $studyReceipt){$previous=Get-Content -LiteralPath $studyReceipt -Raw|ConvertFrom-Json;$studyEvents=@($previous.steps);$studySpent=[double]$previous.active_process_seconds}
if(@($studyEvents|Where-Object{$_.label -eq $Phase}).Count){throw 'This phase has already been attempted; no retry'}
& python (Join-Path $studyPrep 'evidence.py') check-gate
if($LASTEXITCODE -ne 0){throw 'Frozen gate inputs changed'}
if($Phase -ne 'Gate'){
    & python (Join-Path $studyPrep 'evidence.py') check-study
    if($LASTEXITCODE -ne 0){throw 'Frozen study inputs changed'}
}
$remaining=900.0-$studySpent
if($remaining -le 0){throw 'Combined native/process runtime allowance exhausted'}
if($Phase -eq 'Gate'){
    $nativeFile=Join-Path $studyRoot 'build/current-gate-v2-preparation-2026-11-22/build/Release/current_physics_gate_v2.exe'
    $nativeMode='--run';$nativeDirectory=Join-Path $studyRun 'physics-gate'
}else{$nativeFile=Join-Path $studyRoot 'build/Release/ppo_current_swim_v2.exe';$nativeMode='--'+$Phase.ToLowerInvariant();$nativeDirectory=$studyRun}
$monitorProcess=$null
try{
    $monitor=Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if($monitor){
        $monitorProcess=Start-Process -FilePath $monitor.Source -ArgumentList @('--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,power.limit,clocks.current.sm,clocks.current.memory','--format=csv','--loop-ms=1000') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $studyRun ('gpu-telemetry-'+$Phase+'.csv')) -RedirectStandardError (Join-Path $studyRun ('gpu-telemetry-'+$Phase+'.err')) -PassThru
    }
    $clock=[Diagnostics.Stopwatch]::StartNew()
    $nativeProcess=Start-Process -FilePath $nativeFile -ArgumentList @($nativeMode,('"'+$nativeDirectory+'"')) -WorkingDirectory $studyRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $studyRun ($Phase+'.stdout.log')) -RedirectStandardError (Join-Path $studyRun ($Phase+'.stderr.log')) -PassThru
    $studyEvents += [ordered]@{label=$Phase;mode=$nativeMode;pid=$nativeProcess.Id;status='running';seconds=0;exit_code=$null}
    [ordered]@{active_process_seconds=$studySpent;cap_seconds=900;steps=$studyEvents}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $studyReceipt
    $timeout=$false
    while(-not $nativeProcess.WaitForExit(250)){
        if($clock.Elapsed.TotalSeconds -ge $remaining){Stop-Process -Id $nativeProcess.Id -Force;$timeout=$true;break}
    }
    $nativeProcess.WaitForExit();$clock.Stop();$studySpent += $clock.Elapsed.TotalSeconds
    $studyEvents[-1].seconds=$clock.Elapsed.TotalSeconds;$studyEvents[-1].exit_code=$nativeProcess.ExitCode
    $studyEvents[-1].status=if($timeout){'wall_cap_stopped'}elseif($nativeProcess.ExitCode -eq 0){'completed'}else{'failed'}
    [ordered]@{active_process_seconds=$studySpent;cap_seconds=900;steps=$studyEvents}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $studyReceipt
    Get-Content -LiteralPath (Join-Path $studyRun ($Phase+'.stdout.log')) -Tail 12
    Get-Content -LiteralPath (Join-Path $studyRun ($Phase+'.stderr.log')) -Tail 12
    if($timeout){throw 'Combined 900-second process cap reached; no retry'}
    if($nativeProcess.ExitCode -ne 0){throw "Native phase failed: $Phase ($($nativeProcess.ExitCode)); no retry"}
}finally{if($null -ne $monitorProcess -and -not $monitorProcess.HasExited){Stop-Process -Id $monitorProcess.Id}}
& python (Join-Path $studyPrep 'evidence.py') check-gate
if($LASTEXITCODE -ne 0){throw 'Gate inputs changed during the phase'}
if($Phase -ne 'Gate'){
    & python (Join-Path $studyPrep 'evidence.py') check-study
    if($LASTEXITCODE -ne 0){throw 'Study inputs changed during the phase'}
}
