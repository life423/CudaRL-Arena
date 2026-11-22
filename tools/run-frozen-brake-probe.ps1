$ErrorActionPreference='Stop'
$repo='C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation'
$run=Join-Path $repo 'build/diagnostics/frozen-brake-B344-v1'
Set-Location -LiteralPath $repo
if(Test-Path (Join-Path $run 'run-started')){throw 'Physical attempt already started; no retries'}
foreach($h in (Get-Content (Join-Path $run 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $h.path).Hash -ne $h.sha256){throw ('Frozen file changed: '+$h.path)}}
$clock=[Diagnostics.Stopwatch]::StartNew();$start=[DateTime]::UtcNow.ToString('o')
$native=Start-Process -FilePath (Join-Path $run 'runner.exe') -ArgumentList @('--run',('"'+$run+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'native.log') -RedirectStandardError (Join-Path $run 'native.err')
@{pid=$native.Id;started_utc=$start;process_cap_seconds=30}|ConvertTo-Json|Set-Content (Join-Path $run 'launch.json')
Write-Output ('FROZEN_DIAGNOSTIC_STARTED '+$start+' pid='+$native.Id)
$forced=''
while(-not $native.WaitForExit(25)){if($clock.Elapsed.TotalSeconds -ge 30){$forced='Own-process30s cap reached';$native.Kill($true);$native.WaitForExit();break}}
$native.Refresh()
@{pid=$native.Id;started_utc=$start;ended_utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$clock.Elapsed.TotalSeconds;exit_code=$native.ExitCode;forced_stop=$forced}|ConvertTo-Json|Set-Content (Join-Path $run 'execution.json')
foreach($h in (Get-Content (Join-Path $run 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $h.path).Hash -ne $h.sha256){throw ('Frozen input changed during execution: '+$h.path)}}
Get-Content (Join-Path $run 'native.log');Get-Content (Join-Path $run 'native.err')
if(Test-Path (Join-Path $run 'result.json')){Get-Content (Join-Path $run 'result.json')}
if($forced -or $native.ExitCode -ne 0){throw 'Diagnostic closed with failure; preserve data, no retry'}
