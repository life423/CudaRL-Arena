$ErrorActionPreference='Stop'
$repo='C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation'
$study=Join-Path $repo 'build/ppo/balanced-hold-scale-336to344-v1'
Set-Location -LiteralPath $repo
if(Test-Path (Join-Path $study 'run-started')){throw 'Attempt already started; no retries'}
foreach($h in (Get-Content (Join-Path $study 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $h.path).Hash -ne $h.sha256){throw ('Frozen file changed: '+$h.path)}}
$watch=[Diagnostics.Stopwatch]::StartNew();$utc=[DateTime]::UtcNow.ToString('o')
$process=Start-Process -FilePath (Join-Path $study 'runner.exe') -ArgumentList @('--run',('"'+$study+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $study 'native.log') -RedirectStandardError (Join-Path $study 'native.err')
@{started_utc=$utc;pid=$process.Id;process_cap_seconds=360;arm_cap_seconds=180}|ConvertTo-Json|Set-Content (Join-Path $study 'launch.json')
Write-Output ('NATIVE_STARTED '+$utc+' pid='+$process.Id)
$reason=''
while(-not $process.WaitForExit(100)){
 if($watch.Elapsed.TotalSeconds -ge 360){$reason='Combined own-process 360s cap reached';break}
 $active=$null;try{$active=Get-Content (Join-Path $study 'active-arm.json') -Raw|ConvertFrom-Json}catch{}
 if($active){$now=([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0);if(($active.elapsed_previous+$now-$active.phase_started_unix) -ge 180){$reason=('Arm '+$active.arm+' own-process 180s cap reached');break}}
}
if($reason){$process.Kill($true);$process.WaitForExit();@{fault=$reason;no_retry=$true}|ConvertTo-Json|Set-Content (Join-Path $study 'process-stop.json')}
$process.Refresh()
@{started_utc=$utc;ended_utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$watch.Elapsed.TotalSeconds;exit_code=$process.ExitCode;forced_stop=$reason;pid=$process.Id}|ConvertTo-Json|Set-Content (Join-Path $study 'execution.json')
foreach($h in (Get-Content (Join-Path $study 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $h.path).Hash -ne $h.sha256){throw ('Frozen file changed during run: '+$h.path)}}
Get-Content (Join-Path $study 'native.log') -Tail 28
Get-Content (Join-Path $study 'native.err')
if(Test-Path (Join-Path $study 'result.json')){Get-Content (Join-Path $study 'result.json')}
if($process.ExitCode -ne 0 -or $reason){throw 'Bounded study stopped; preserve evidence, no retries'}
