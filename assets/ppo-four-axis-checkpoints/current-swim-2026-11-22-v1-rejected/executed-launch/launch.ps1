param([Parameter(Mandatory=$true)][ValidateSet('GateAndCalibrate','Train')][string]$Phase)
$ErrorActionPreference = 'Stop'
$studyPrep = $PSScriptRoot
$studyRoot = [IO.Path]::GetFullPath((Join-Path $studyPrep '../..'))
$studyRun = Join-Path $studyRoot 'build/ppo/current-swim-2026-11-22-v1'
$studyGate = Join-Path $studyRun 'physics-gate'
$studyReceipt = Join-Path $studyRun 'process-execution.json'
$studyEvents = @()
$studyActiveSeconds = 0.0
if (Test-Path -LiteralPath $studyReceipt) {
    $previousExecution = Get-Content -LiteralPath $studyReceipt -Raw | ConvertFrom-Json
    $studyEvents = @($previousExecution.steps)
    $studyActiveSeconds = [double]$previousExecution.active_process_seconds
}
function Save-StudyExecution {
    [ordered]@{active_process_seconds=$script:studyActiveSeconds;cap_seconds=900;steps=$script:studyEvents} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $script:studyReceipt
}
function Run-StudyNative([string]$Label,[string]$Executable,[string]$Mode,[string]$Directory) {
    if (@($script:studyEvents | Where-Object { $_.label -eq $Label }).Count -ne 0) { throw "Phase already attempted: $Label" }
    & python (Join-Path $script:studyPrep 'evidence.py') check-frozen
    if ($LASTEXITCODE -ne 0) { throw 'Frozen execution inputs changed' }
    $remainingSeconds = 900.0 - $script:studyActiveSeconds
    if ($remainingSeconds -le 0) { throw 'Combined process budget exhausted' }
    $outputLog = Join-Path $script:studyRun ($Label + '.stdout.log')
    $errorLog = Join-Path $script:studyRun ($Label + '.stderr.log')
    $stepClock = [Diagnostics.Stopwatch]::StartNew()
    $nativeProcess = Start-Process -FilePath $Executable -ArgumentList @($Mode,('"'+$Directory+'"')) -WorkingDirectory $script:studyRoot -WindowStyle Hidden -RedirectStandardOutput $outputLog -RedirectStandardError $errorLog -PassThru
    $script:studyEvents += [ordered]@{label=$Label;mode=$Mode;pid=$nativeProcess.Id;status='running';seconds=0;exit_code=$null}
    Save-StudyExecution
    $timedOut = $false
    while (-not $nativeProcess.WaitForExit(250)) {
        if ($stepClock.Elapsed.TotalSeconds -ge $remainingSeconds) {
            Stop-Process -Id $nativeProcess.Id -Force
            $timedOut = $true
            break
        }
    }
    $nativeProcess.WaitForExit()
    $stepClock.Stop()
    $script:studyActiveSeconds += $stepClock.Elapsed.TotalSeconds
    $script:studyEvents[-1].seconds = $stepClock.Elapsed.TotalSeconds
    $script:studyEvents[-1].exit_code = $nativeProcess.ExitCode
    $script:studyEvents[-1].status = if ($timedOut) {'wall_cap_stopped'} elseif ($nativeProcess.ExitCode -eq 0) {'completed'} else {'failed'}
    Save-StudyExecution
    Get-Content -LiteralPath $outputLog -Tail 8
    Get-Content -LiteralPath $errorLog -Tail 8
    if ($timedOut) { throw 'Combined 900-second process cap reached; no retry' }
    if ($nativeProcess.ExitCode -ne 0) { throw "Native phase failed: $Label ($($nativeProcess.ExitCode)); no retry" }
    & python (Join-Path $script:studyPrep 'evidence.py') check-frozen
    if ($LASTEXITCODE -ne 0) { throw 'Execution inputs changed during the phase' }
}
if ($Phase -eq 'GateAndCalibrate') {
    if ($studyEvents.Count -ne 0) { throw 'Gate/calibration identity has already been attempted' }
    Run-StudyNative 'physical-gate' (Join-Path $studyRoot 'build/Release/current_physics_gate.exe') '--run' $studyGate
    $gateResult = Get-Content -LiteralPath (Join-Path $studyGate 'run-started/result.json') -Raw | ConvertFrom-Json
    if (-not $gateResult.passed -or $gateResult.attempted_controls -ne 3760 -or $gateResult.physical_substeps -ne 37600) { throw 'Physical gate did not pass exactly; no calibration authorized' }
    Run-StudyNative 'calibration' (Join-Path $studyRoot 'build/Release/ppo_current_swim.exe') '--calibrate' $studyRun
} else {
    if ($studyEvents.Count -ne 2 -or -not (Test-Path -LiteralPath (Join-Path $studyRun 'calibration-ready.bin'))) { throw 'One completed, qualified calibration is required before training' }
    Run-StudyNative 'training' (Join-Path $studyRoot 'build/Release/ppo_current_swim.exe') '--train' $studyRun
}
