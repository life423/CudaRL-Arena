param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Gate','Baseline','Calibrate','Train')]
    [string]$Phase
)
$ErrorActionPreference='Stop'
$taskRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if((Split-Path $taskRoot -Leaf) -ne 'minimal-foundation') { throw 'Wrong active worktree' }
$runRoot=Join-Path $taskRoot 'build/ppo/continuous-pond-v2-2026-11-23-v1'
$ledgerPath=Join-Path $runRoot 'process-execution.json'
if(-not (Test-Path -LiteralPath (Join-Path $runRoot 'authorization.bin'))) { throw 'Fresh user approval is required before physical execution' }
$records=@()
if(Test-Path -LiteralPath $ledgerPath) { $records=@(Get-Content -LiteralPath $ledgerPath -Raw | ConvertFrom-Json) }
if(@($records | Where-Object phase -eq $Phase).Count) { throw 'A physical phase may not be repeated' }
$used=0.0
foreach($record in $records) { $used += $record.seconds }
$remaining=900.0-$used
if($remaining -le 0) { throw 'The finite 900-second combined native process budget is closed' }
if((Get-PSDrive C).Free -lt 8GB) { throw 'The protected disk reserve is unavailable' }
if(-not (Test-Path -LiteralPath (Join-Path $runRoot 'frozen-inputs.json'))) { throw 'Freeze inputs before physical execution' }
$executable=Join-Path $taskRoot 'build/continuous-pond-v2-native-2026-11-23/Release/ppo_turning_swim_v2.exe'
$arguments=@("--$($Phase.ToLowerInvariant())",$runRoot)
if($Phase -eq 'Gate') {
    $executable=Join-Path $taskRoot 'build/continuous-pond-v2-native-2026-11-23/Release/turning_physics_gate_v2.exe'
    $arguments=@($runRoot)
}
$telemetryPath=Join-Path $runRoot "gpu-telemetry-$Phase.csv"
$stdout=Join-Path $runRoot "$Phase.stdout.log"
$stderr=Join-Path $runRoot "$Phase.stderr.log"
$watch=[Diagnostics.Stopwatch]::StartNew()
$process=Start-Process -FilePath $executable -ArgumentList ($arguments | ForEach-Object { '"'+$_+'"' }) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$killed=$false
while(-not $process.HasExited) {
    if($watch.Elapsed.TotalSeconds -ge $remaining) { $process.Kill(); $killed=$true; break }
    & nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits | Add-Content -LiteralPath $telemetryPath
    $process.WaitForExit(900) | Out-Null
    $process.Refresh()
}
$process.WaitForExit()
$watch.Stop()
$record=[ordered]@{phase=$Phase;seconds=$watch.Elapsed.TotalSeconds;exit_code=$process.ExitCode;terminated_at_cap=$killed;executable=$executable;executable_sha256=(Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant();stdout=$stdout;stderr=$stderr}
$records+=,$record
ConvertTo-Json -InputObject @($records) -Depth 6 | Set-Content -LiteralPath $ledgerPath -Encoding utf8
Get-Content -LiteralPath $stdout -Tail 6
Get-Content -LiteralPath $stderr -Tail 6
ConvertTo-Json -InputObject $record -Compress
if($killed -or $process.ExitCode -ne 0) { exit 3 }
