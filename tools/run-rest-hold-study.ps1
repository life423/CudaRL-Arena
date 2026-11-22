$ErrorActionPreference='Stop'
$repo='C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation'
$run=Join-Path $repo 'build/ppo/rest-hold-336to344-v1'
Set-Location -LiteralPath $repo
if(Test-Path (Join-Path $run 'run-started')){throw 'Physical attempt already started; no retries'}
foreach($h in (Get-Content (Join-Path $run 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $h.path).Hash -ne $h.sha256){throw ('Frozen input changed: '+$h.path)}}
$clock=[Diagnostics.Stopwatch]::StartNew();$start=[DateTime]::UtcNow.ToString('o')
$native=Start-Process -FilePath (Join-Path $run 'runner.exe') -ArgumentList @('--run',('"'+$run+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'native.log') -RedirectStandardError (Join-Path $run 'native.err')
@{pid=$native.Id;started_utc=$start;process_cap_seconds=120}|ConvertTo-Json|Set-Content (Join-Path $run 'launch.json')
Write-Output ('REST_NATIVE_STARTED '+$start+' pid='+$native.Id)
$forced='';$reported=$false
while(-not $native.WaitForExit(50)){
 if(-not $reported -and (Test-Path (Join-Path $run 'awaiting-report-release'))){$reported=$true;Write-Output 'CALIBRATION_REPORT_READY_NATIVE_PAUSED_BEFORE_PPO'}
 if($clock.Elapsed.TotalSeconds -ge 120){$forced='Own-process120s cap reached';$native.Kill($true);$native.WaitForExit();break}
}
$native.Refresh();@{pid=$native.Id;started_utc=$start;ended_utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$clock.Elapsed.TotalSeconds;exit_code=$native.ExitCode;forced_stop=$forced}|ConvertTo-Json|Set-Content (Join-Path $run 'execution.json')
foreach($h in (Get-Content (Join-Path $run 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $h.path).Hash -ne $h.sha256){throw ('Frozen input changed during run: '+$h.path)}}
Get-Content (Join-Path $run 'native.log') -Tail 28;Get-Content (Join-Path $run 'native.err')
if(Test-Path (Join-Path $run 'pre-PPO-report.json')){Get-Content (Join-Path $run 'pre-PPO-report.json')}
if(Test-Path (Join-Path $run 'result.json')){Get-Content (Join-Path $run 'result.json')}
if($forced -or $native.ExitCode -ne 0){throw 'Rest trial closed with hard failure; no retry'}
