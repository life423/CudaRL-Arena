$ErrorActionPreference='Stop'
$repo='C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation'
Set-Location -LiteralPath $repo
$prep=Join-Path $repo 'build/review-preservation/2026-11-04-before-frozen-brake'
$clock=[Diagnostics.Stopwatch]::StartNew()
$native=Start-Process -FilePath (Get-Command cmake).Source -ArgumentList @('--build','build','--config','Release','--target','frozen_brake_probe','--parallel','2') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $prep 'build.log') -RedirectStandardError (Join-Path $prep 'build.err')
while(-not $native.WaitForExit(100)){if($clock.Elapsed.TotalSeconds -ge 180){$native.Kill($true);throw '180s build preparation cap exhausted; no physical run'}}
$native.Refresh();@{seconds=$clock.Elapsed.TotalSeconds;exit_code=$native.ExitCode}|ConvertTo-Json|Set-Content (Join-Path $prep 'build-result.json')
Get-Content (Join-Path $prep 'build.log') -Tail 24;Get-Content (Join-Path $prep 'build.err') -Tail 16
if($native.ExitCode -ne 0){throw 'Frozen diagnostic build failed; no physical run'}
