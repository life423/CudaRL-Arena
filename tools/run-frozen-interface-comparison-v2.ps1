param([Parameter(Mandatory=$true)] [string]$RunRoot)
$ErrorActionPreference='Stop'
$comparisonWorkspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if((Split-Path $comparisonWorkspace -Leaf) -ne 'minimal-foundation') { throw 'Wrong active worktree' }
$comparisonRun=[IO.Path]::GetFullPath($RunRoot)
if((Split-Path $comparisonRun -Parent) -ne (Join-Path $comparisonWorkspace 'build\ppo')) { throw 'Wrong comparison evidence directory' }
foreach($comparisonStop in @('design-hold.json','process-execution.json','probe-started','result.json')) {
    if(Test-Path -LiteralPath (Join-Path $comparisonRun $comparisonStop)) { throw "Comparison held or already attempted: $comparisonStop" }
}
foreach($comparisonRequired in @('authorization.bin','authorization.json','frozen-inputs.json','independent-preparation-verification.json')) {
    if(-not (Test-Path -LiteralPath (Join-Path $comparisonRun $comparisonRequired))) { throw "Missing $comparisonRequired" }
}
$comparisonProof=Get-Content -LiteralPath (Join-Path $comparisonRun 'independent-preparation-verification.json') -Raw | ConvertFrom-Json
if(-not $comparisonProof.passed -or $comparisonProof.physical_run_approved -or $comparisonProof.physical_controls -ne 0 -or $comparisonProof.optimizer_steps -ne 0) { throw 'Invalid preparation proof' }
foreach($comparisonInput in $comparisonProof.input_sha256.PSObject.Properties) {
    if((Get-FileHash -LiteralPath $comparisonInput.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $comparisonInput.Value.ToLowerInvariant()) { throw "Prepared input changed: $($comparisonInput.Name)" }
}
foreach($comparisonInput in (Get-Content -LiteralPath (Join-Path $comparisonRun 'frozen-inputs.json') -Raw | ConvertFrom-Json)) {
    if((Get-FileHash -LiteralPath $comparisonInput.path).Hash.ToLowerInvariant() -ne $comparisonInput.sha256) { throw "Frozen input changed: $($comparisonInput.path)" }
}
$comparisonExe=Join-Path $comparisonWorkspace 'build/frozen-interface-comparison-v2-native-2026-11-23/Release/frozen_interface_comparison_v2.exe'
$comparisonWatch=[Diagnostics.Stopwatch]::StartNew()
$comparisonProcess=Start-Process -FilePath $comparisonExe -ArgumentList @('"'+$comparisonRun+'"') -WorkingDirectory $comparisonWorkspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $comparisonRun 'execution.stdout.log') -RedirectStandardError (Join-Path $comparisonRun 'execution.stderr.log')
$comparisonKilled=$false
while(-not $comparisonProcess.HasExited) {
    if($comparisonWatch.Elapsed.TotalSeconds -ge 900) { $comparisonProcess.Kill(); $comparisonKilled=$true; break }
    $comparisonProcess.WaitForExit(250) | Out-Null
    $comparisonProcess.Refresh()
}
$comparisonProcess.WaitForExit()
$comparisonWatch.Stop()
$comparisonRecord=[ordered]@{schema=1;study='frozen_interface_comparison_v2';seconds=$comparisonWatch.Elapsed.TotalSeconds;exit_code=$comparisonProcess.ExitCode;terminated_at_cap=$comparisonKilled;executable=$comparisonExe;executable_sha256=(Get-FileHash -LiteralPath $comparisonExe -Algorithm SHA256).Hash.ToLowerInvariant();PPO_updates=0;optimizer_steps=0}
ConvertTo-Json -InputObject $comparisonRecord -Depth 4 | Set-Content -LiteralPath (Join-Path $comparisonRun 'process-execution.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $comparisonRun 'execution.stdout.log') -Tail 4
Get-Content -LiteralPath (Join-Path $comparisonRun 'execution.stderr.log') -Tail 4
$comparisonRecord | ConvertTo-Json -Compress
if($comparisonKilled -or $comparisonProcess.ExitCode -ne 0) { exit 3 }
