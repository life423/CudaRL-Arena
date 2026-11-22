$ErrorActionPreference='Stop'
$repo='C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation'
Set-Location -LiteralPath $repo
$prep=Join-Path $repo 'build/review-preservation/2026-11-04-before-balanced-scale'
$cmake=(Get-Command cmake).Source
$clock=[Diagnostics.Stopwatch]::StartNew()
$buildProcess=Start-Process -FilePath $cmake -ArgumentList @('--build','build','--config','Release','--target','ppo_hold_scale','ppo_append_hold','ppo_motor','foundation_tests','--parallel','2') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $prep 'build.log') -RedirectStandardError (Join-Path $prep 'build.err')
while(-not $buildProcess.WaitForExit(100)){if($clock.Elapsed.TotalSeconds -ge 300){$buildProcess.Kill($true);throw '300s build cap reached; no physical run attempted'}}
$buildProcess.Refresh()
@{seconds=$clock.Elapsed.TotalSeconds;exit_code=$buildProcess.ExitCode}|ConvertTo-Json|Set-Content (Join-Path $prep 'build-result.json')
if($buildProcess.ExitCode -ne 0){Get-Content (Join-Path $prep 'build.log') -Tail 65;Get-Content (Join-Path $prep 'build.err') -Tail 25;throw 'Build failed; no physical run attempted'}
Get-Content (Join-Path $prep 'build.log') -Tail 16
