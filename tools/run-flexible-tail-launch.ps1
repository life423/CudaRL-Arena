param(
    [Parameter(Mandatory)][string]$BinaryDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$binary = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
$fixture = Join-Path $binary 'pond_flexible_tail_motor_launch_fixture.exe'
$trainer = Join-Path $binary 'pond_flexible_tail_motor_navigation_ppo.exe'
$saved = Join-Path $repository 'build/flexible-tail/continuous-turn-ppo-v1/capture'
$base = Join-Path $repository 'build/flexible-tail/motor-pilot-v1/actor.bin'
$projectSources = Join-Path $PSScriptRoot 'pond-flexible-tail-motor'
$protocol = Join-Path $projectSources 'LAUNCH_PROTOCOL.md'
$protected = [ordered]@{
    $base = 'AC87BD493710DAACA878AA52CAF3A0D701EC879596AC44417235AC2979E42EDF'
    (Join-Path $saved 'train-worlds.bin') = 'C887927143DAFA3718A18649897C22D74B32FE01A2D8A765C77336EDFCB4A0A9'
    (Join-Path $saved 'eval-worlds.bin') = '1A067E2F11D6E36352FCA2B56D19D5D7602B4D0C4A7693711187E15A438FAAA1'
}
$capMilliseconds = 4000

function Assert-ProtectedInputs {
    foreach ($file in $protected.Keys) {
        if ((Get-FileHash -LiteralPath $file).Hash -ne $protected[$file]) {
            throw "Original launch input hash differs for $file"
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

    $clock = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::Start($info)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $remaining = [Math]::Max(0, $capMilliseconds - $clock.ElapsedMilliseconds)
        $finished = $process.WaitForExit([int]$remaining)
        $timedOut = !$finished -or $clock.Elapsed.TotalMilliseconds -gt $capMilliseconds
        if ($timedOut -and !$process.HasExited) {
            $process.Kill($true)
            if (!$process.WaitForExit(500)) { throw "Could not stop $Name after timeout" }
        }
        $clock.Stop()
        $record = [ordered]@{
            stage = $Name
            executable = $Executable
            arguments = $Arguments
            exit_code = $process.ExitCode
            timed_out = $timedOut
            wall_s = $clock.Elapsed.TotalSeconds
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
$inputs = @($trainer, $fixture, $protocol, $PSCommandPath,
    (Join-Path $projectSources 'launch-fixture.cpp'), (Join-Path $projectSources 'navigation-ppo.cu'),
    (Join-Path $projectSources 'navigation-kernels.cu'), (Join-Path $projectSources 'navigation-kernels.h'),
    (Join-Path $repository 'src/rl/flexible_tail_launch.h'),
    (Join-Path $repository 'src/rl/flexible_tail_navigation.h'),
    (Join-Path $repository 'src/rl/flexible_tail_transition.h'),
    (Join-Path $repository 'src/physics/flexible_tail.h'),
    (Join-Path $repository 'flexible-tail-preview/habitat.json')) + @($protected.Keys)
$bindings = Get-FileHash -LiteralPath $inputs
if (Test-Path -LiteralPath $output) { throw "Study output already exists at $output" }
$null = New-Item -Path $output -ItemType Directory
$studyClock = [Diagnostics.Stopwatch]::StartNew()
$bindings | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'bindings.json')
Copy-Item -LiteralPath $protocol -Destination (Join-Path $output 'protocol.md')
$capture = Join-Path $output 'capture'
$training = Join-Path $output 'training'
$checked = [System.Collections.Generic.List[int]]::new()
$trained = $false
try {
    if (!(Invoke-BoundedStage 'capture' $fixture @('capture', $saved, $capture))) {
        throw 'Fixture preparation failed before training'
    }
    $trainWorlds = Join-Path $capture 'train-worlds.bin'
    $evalWorlds = Join-Path $capture 'eval-worlds.bin'
    Get-FileHash -LiteralPath $trainWorlds, $evalWorlds | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $output 'fixture-hashes.json')
    $trained = Invoke-BoundedStage 'train' $trainer @('launch', $base, $trainWorlds, $training)
    foreach ($update in @(0, 4, 8, 12)) {
        $actor = Join-Path $training "actor-$update.bin"
        if (!(Test-Path -LiteralPath $actor)) { break }
        if (!(Invoke-BoundedStage "check-$update" $fixture @('evaluate', $actor, $evalWorlds,
            (Join-Path $output "check-$update")))) {
            throw "Assessment stopped at update $update"
        }
        $checked.Add($update)
    }
    if (!$trained -or $checked.Count -ne 4) { throw 'Pilot stopped early. See the saved receipts.' }
} finally {
    $studyClock.Stop()
    [ordered]@{
        process_cap_s = $capMilliseconds / 1000
        trainer_completed = $trained
        evaluated_updates = @($checked)
        wall_s = $studyClock.Elapsed.TotalSeconds
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output 'study.json')
    Assert-ProtectedInputs
}
