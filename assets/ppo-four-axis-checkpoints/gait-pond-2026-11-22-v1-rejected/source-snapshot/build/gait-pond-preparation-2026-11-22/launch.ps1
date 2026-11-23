param([Parameter(Mandatory=$true)][ValidateSet('Gate','Baseline','Calibrate','Train')][string]$Phase)
$ErrorActionPreference='Stop'
$gaitPrep=$PSScriptRoot
$gaitRoot=[IO.Path]::GetFullPath((Join-Path $gaitPrep '../..'))
$gaitRun=Join-Path $gaitRoot 'build/ppo/gait-pond-2026-11-22-v1'
$gaitReceipt=Join-Path $gaitRun 'process-execution.json'
$gaitEvents=@();$gaitSpent=0.0
if(Test-Path -LiteralPath $gaitReceipt){$previous=Get-Content -LiteralPath $gaitReceipt -Raw|ConvertFrom-Json;$gaitEvents=@($previous.steps);$gaitSpent=[double]$previous.active_process_seconds}
if(@($gaitEvents|Where-Object{$_.label -eq $Phase}).Count){throw 'This phase was already attempted; no retry'}
& python (Join-Path $gaitPrep 'evidence.py') check
if($LASTEXITCODE -ne 0){throw 'Frozen gait inputs changed'}
$gaitRemaining=900.0-$gaitSpent
if($gaitRemaining -le 0){throw 'Combined runtime allowance exhausted'}
$gaitNative=Join-Path $gaitRoot 'build/gait-pond-native-2026-11-22/Release/ppo_gait_swim.exe'
$gaitMode='--'+$Phase.ToLowerInvariant()
$gaitMonitor=$null
try {
    $gaitSmi=Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if($gaitSmi){$gaitMonitor=Start-Process -FilePath $gaitSmi.Source -ArgumentList @('--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,power.limit','--format=csv','--loop-ms=1000') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $gaitRun ('gpu-telemetry-'+$Phase+'.csv')) -RedirectStandardError (Join-Path $gaitRun ('gpu-telemetry-'+$Phase+'.err')) -PassThru}
    $gaitClock=[Diagnostics.Stopwatch]::StartNew()
    $gaitProcess=Start-Process -FilePath $gaitNative -ArgumentList @($gaitMode,('"'+$gaitRun+'"')) -WorkingDirectory $gaitRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $gaitRun ($Phase+'.stdout.log')) -RedirectStandardError (Join-Path $gaitRun ($Phase+'.stderr.log')) -PassThru
    $gaitEvents += [ordered]@{label=$Phase;mode=$gaitMode;pid=$gaitProcess.Id;status='running';seconds=0;exit_code=$null}
    [ordered]@{active_process_seconds=$gaitSpent;cap_seconds=900;steps=$gaitEvents}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $gaitReceipt
    $gaitTimeout=$false
    while(-not $gaitProcess.WaitForExit(250)){if($gaitClock.Elapsed.TotalSeconds -ge $gaitRemaining){Stop-Process -Id $gaitProcess.Id -Force;$gaitTimeout=$true;break}}
    $gaitProcess.WaitForExit();$gaitClock.Stop();$gaitSpent += $gaitClock.Elapsed.TotalSeconds
    $gaitEvents[-1].seconds=$gaitClock.Elapsed.TotalSeconds;$gaitEvents[-1].exit_code=$gaitProcess.ExitCode
    $gaitEvents[-1].status=if($gaitTimeout){'wall_cap_stopped'}elseif($gaitProcess.ExitCode -eq 0){'completed'}else{'failed'}
    [ordered]@{active_process_seconds=$gaitSpent;cap_seconds=900;steps=$gaitEvents}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $gaitReceipt
    Get-Content -LiteralPath (Join-Path $gaitRun ($Phase+'.stdout.log')) -Tail 8
    Get-Content -LiteralPath (Join-Path $gaitRun ($Phase+'.stderr.log')) -Tail 8
    if($gaitTimeout){throw '900-second combined process cap reached; no retry'}
    if($gaitProcess.ExitCode -ne 0){throw "Native gait phase failed: $Phase; no retry"}
} finally {if($null -ne $gaitMonitor -and -not $gaitMonitor.HasExited){Stop-Process -Id $gaitMonitor.Id}}
& python (Join-Path $gaitPrep 'evidence.py') check
if($LASTEXITCODE -ne 0){throw 'Gait causal inputs changed during execution'}
