param(
    [Parameter(Mandatory)][string]$BinaryDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$binary = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
$study = Join-Path $PSScriptRoot 'pond-flexible-tail-motor'
$fixture = Join-Path $binary 'pond_flexible_tail_motor_navigation_fixture.exe'
$trainer = Join-Path $binary 'pond_flexible_tail_motor_navigation_ppo.exe'
$saved = Join-Path $repository 'build/flexible-tail/continuous-turn-ppo-v1/capture'
$base = Join-Path $repository 'build/flexible-tail/motor-pilot-v1/actor.bin'
$protected = [ordered]@{
    $base = 'AC87BD493710DAACA878AA52CAF3A0D701EC879596AC44417235AC2979E42EDF'
    (Join-Path $saved 'train-worlds.bin') = 'C887927143DAFA3718A18649897C22D74B32FE01A2D8A765C77336EDFCB4A0A9'
    (Join-Path $saved 'eval-worlds.bin') = '1A067E2F11D6E36352FCA2B56D19D5D7602B4D0C4A7693711187E15A438FAAA1'
}
$maximumMilliseconds = 4000
$checkpoints = @(0, 2, 4, 6, 8)
$modes = @('heading', 'destination')

function Assert-ProtectedInputs {
    foreach ($file in $protected.Keys) {
        if ((Get-FileHash -LiteralPath $file).Hash -ne $protected[$file]) {
            throw "Saved input hash differs for $file"
        }
    }
}

function Invoke-BoundedStage([string]$Name, [string]$Executable, [string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.WorkingDirectory = $repository
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }

    $timer = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::Start($info)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $remaining = [Math]::Max(0, $maximumMilliseconds - $timer.ElapsedMilliseconds)
        $timedOut = !$process.WaitForExit([int]$remaining) -or $timer.Elapsed.TotalMilliseconds -gt $maximumMilliseconds
        if ($timedOut -and !$process.HasExited) {
            $process.Kill($true)
            if (!$process.WaitForExit(500)) { throw "Could not stop $Name after timeout" }
        }
        $timer.Stop()
        $record = [ordered]@{
            stage = $Name
            command = $Executable
            arguments = $Arguments
            exit_code = $process.ExitCode
            timed_out = $timedOut
            wall_s = $timer.Elapsed.TotalSeconds
        }
        Set-Content -LiteralPath (Join-Path $output "$Name.stdout.log") -Value ($stdout.GetAwaiter().GetResult())
        Set-Content -LiteralPath (Join-Path $output "$Name.stderr.log") -Value ($stderr.GetAwaiter().GetResult())
        $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output "$Name.receipt.json")
        Write-Host ("{0} completed in {1:F3} s, exit {2}, timeout {3}" -f $Name, $record.wall_s,
            $record.exit_code, $record.timed_out)
        return (!$timedOut -and $process.ExitCode -eq 0)
    } finally {
        $process.Dispose()
    }
}

Assert-ProtectedInputs
$inputs = @($trainer, $fixture, (Join-Path $study 'NAVIGATION_COMPARISON.md'),
    (Join-Path $study 'navigation-fixture.cpp'), (Join-Path $study 'navigation-ppo.cu'),
    (Join-Path $study 'navigation-kernels.cu'),
    (Join-Path $repository 'src/rl/flexible_tail_navigation.h'),
    (Join-Path $repository 'src/rl/flexible_tail_transition.h'),
    (Join-Path $repository 'src/physics/flexible_tail.h'),
    (Join-Path $repository 'flexible-tail-preview/habitat.json'), $PSCommandPath) + @($protected.Keys)
$bindings = Get-FileHash -LiteralPath $inputs
if (Test-Path -LiteralPath $output) { throw "Study output already exists at $output" }
$null = New-Item -Path $output -ItemType Directory
$timer = [Diagnostics.Stopwatch]::StartNew()
$bindings | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'bindings.json')
Copy-Item -LiteralPath (Join-Path $study 'NAVIGATION_COMPARISON.md') -Destination (Join-Path $output 'protocol.md')
$capture = Join-Path $output 'capture'
$checked = [System.Collections.Generic.List[int]]::new()
$completion = [ordered]@{}
try {
    if (!(Invoke-BoundedStage 'capture' $fixture @('capture', $saved, $capture))) {
        throw 'Fixture preparation failed. Training did not start.'
    }
    $trainWorlds = Join-Path $capture 'train-worlds.bin'
    $evalWorlds = Join-Path $capture 'eval-worlds.bin'
    Get-FileHash -LiteralPath $trainWorlds, $evalWorlds | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $output 'fixture-hashes.json')
    foreach ($mode in $modes) {
        $completion[$mode] = Invoke-BoundedStage "train-$mode" $trainer @($mode, $base, $trainWorlds,
            (Join-Path $output $mode))
    }
    foreach ($update in $checkpoints) {
        $heading = Join-Path $output "heading/actor-$update.bin"
        $destination = Join-Path $output "destination/actor-$update.bin"
        if (!(Test-Path -LiteralPath $heading) -or !(Test-Path -LiteralPath $destination)) { break }
        if (!(Invoke-BoundedStage "check-$update" $fixture @('evaluate', $heading, $destination, $evalWorlds,
            (Join-Path $output "check-$update")))) {
            throw "Paired evaluation stopped at update $update"
        }
        $checked.Add($update)
    }
    if ($checked.Count -ne $checkpoints.Count -or $completion.Values -contains $false) {
        throw 'The paired study stopped early. Completed stages are recorded.'
    }
} finally {
    $timer.Stop()
    $studyRecord = [ordered]@{
        mode_order = $modes
        process_cap_s = $maximumMilliseconds / 1000
        trainer_pass = $completion
        evaluated_updates = @($checked)
        wall_s = $timer.Elapsed.TotalSeconds
    }
    $studyRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output 'study.json')
    Assert-ProtectedInputs
}
