param(
    [Parameter(Mandatory=$true)] [string]$RunRoot
)
$ErrorActionPreference='Stop'
$probeWorkspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if((Split-Path $probeWorkspace -Leaf) -ne 'minimal-foundation') { throw 'Wrong active worktree' }
$probeRun=[IO.Path]::GetFullPath($RunRoot)
$probeRunsPrefix=[IO.Path]::GetFullPath((Join-Path $probeWorkspace 'build/ppo'))+[IO.Path]::DirectorySeparatorChar
if(-not $probeRun.StartsWith($probeRunsPrefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Probe root is outside the intended workspace area' }
foreach($probeRequired in @('authorization.bin','authorization.json','independent-preparation-verification.json','frozen-inputs.json')) {
    if(-not (Test-Path -LiteralPath (Join-Path $probeRun $probeRequired))) { throw "Missing required probe evidence: $probeRequired" }
}
foreach($probeStop in @('design-hold.json','process-execution.json','probe-started','result.json')) {
    if(Test-Path -LiteralPath (Join-Path $probeRun $probeStop)) { throw "Probe held or already attempted: $probeStop" }
}
$probeProof=Get-Content -LiteralPath (Join-Path $probeRun 'independent-preparation-verification.json') -Raw | ConvertFrom-Json
if(-not $probeProof.passed -or $probeProof.physical_run_approved -or $probeProof.physical_controls -ne 0 -or $probeProof.optimizer_steps -ne 0) { throw 'Probe preparation is not valid' }
foreach($probeInput in $probeProof.input_sha256.PSObject.Properties) {
    if((Get-FileHash -LiteralPath $probeInput.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $probeInput.Value.ToLowerInvariant()) { throw "Prepared input changed: $($probeInput.Name)" }
}
$probeExe=Join-Path $probeWorkspace 'build/frozen-request-turn-native-2026-11-23/Release/frozen_requested_turn_probe.exe'
$probeWatch=[Diagnostics.Stopwatch]::StartNew()
$probeProcess=Start-Process -FilePath $probeExe -ArgumentList @('"'+$probeRun+'"') -WorkingDirectory $probeWorkspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $probeRun 'execution.stdout.log') -RedirectStandardError (Join-Path $probeRun 'execution.stderr.log')
$probeKilled=$false
while(-not $probeProcess.HasExited) {
    if($probeWatch.Elapsed.TotalSeconds -ge 120) { $probeProcess.Kill(); $probeKilled=$true; break }
    $probeProcess.WaitForExit(250) | Out-Null
    $probeProcess.Refresh()
}
$probeProcess.WaitForExit()
$probeWatch.Stop()
$probeRecord=[ordered]@{schema=1;study='frozen_requested_turn_probe';seconds=$probeWatch.Elapsed.TotalSeconds;exit_code=$probeProcess.ExitCode;terminated_at_cap=$probeKilled;executable=$probeExe;executable_sha256=(Get-FileHash -LiteralPath $probeExe -Algorithm SHA256).Hash.ToLowerInvariant();PPO_updates=0;optimizer_steps=0}
ConvertTo-Json -InputObject $probeRecord -Depth 4 | Set-Content -LiteralPath (Join-Path $probeRun 'process-execution.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $probeRun 'execution.stdout.log') -Tail 4
Get-Content -LiteralPath (Join-Path $probeRun 'execution.stderr.log') -Tail 4
$probeRecord | ConvertTo-Json -Compress
if($probeKilled -or $probeProcess.ExitCode -ne 0) { exit 3 }
