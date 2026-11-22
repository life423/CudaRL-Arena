$ErrorActionPreference='Stop'
$repo='C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation'
Set-Location -LiteralPath $repo
$prep=Join-Path $repo 'build/review-preservation/2026-11-04-before-rest-hold'
$clock=[Diagnostics.Stopwatch]::StartNew()
$native=Start-Process -FilePath (Get-Command cmake).Source -ArgumentList @('--build','build','--config','Release','--target','ppo_rest_hold','foundation_tests','ppo_motor','ppo_append_hold','ppo_hold_scale','frozen_brake_probe','--parallel','2') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $prep 'build.log') -RedirectStandardError (Join-Path $prep 'build.err')
while(-not $native.WaitForExit(100)){if($clock.Elapsed.TotalSeconds -ge 300){$native.Kill($true);throw '300s preparation build cap exhausted; no physical attempt'}}
$native.Refresh();@{seconds=$clock.Elapsed.TotalSeconds;exit_code=$native.ExitCode}|ConvertTo-Json|Set-Content (Join-Path $prep 'build-result.json')
Get-Content (Join-Path $prep 'build.log') -Tail 35;Get-Content (Join-Path $prep 'build.err') -Tail 15
if($native.ExitCode -ne 0){throw 'Rest-hold build failed; no physical attempt'}
