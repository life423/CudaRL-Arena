param(
    [Parameter(Mandatory)][string]$SessionDir,
    [Parameter(Mandatory)][ValidateSet('held-heading', 'measured-course')][string]$SteeringMode,
    [switch]$Resume,
    [ValidateRange(1, 64)][int]$TargetUpdates = 64,
    [ValidateRange(60, 1800)][int]$WallSeconds = 1800,
    [string]$BuildDirectory = 'build/flexible-tail/chooser-session-build',
    [string]$ResumeVerificationDir = 'build/flexible-tail/chooser-session-resume-check',
    [string]$SteeringEvidenceDir = 'build/flexible-tail/steering-interface-v3'
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$active = @()
$lease = $null
Push-Location -LiteralPath $repository

function Write-AtomicJson([string]$Path, $Value) {
    $temporary = $Path + '.tmp'
    $Value | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $temporary -Encoding utf8
    [IO.File]::Move($temporary, $Path, $true)
}

function Get-RelativeBindings([string[]]$Paths) {
    foreach ($path in ($Paths | Sort-Object -Unique)) {
        $file = (Resolve-Path -LiteralPath $path).Path
        [pscustomobject]@{
            path = [IO.Path]::GetRelativePath($repository, $file).Replace([IO.Path]::DirectorySeparatorChar, [char]'/')
            sha256 = (Get-FileHash -LiteralPath $file).Hash
        }
    }
}

function Assert-Bindings($Bindings) {
    foreach ($binding in $Bindings) {
        $path = Join-Path $repository $binding.path
        if (!(Test-Path -LiteralPath $path) -or (Get-FileHash -LiteralPath $path).Hash -ne $binding.sha256) {
            throw "The session input changed at $($binding.path)"
        }
    }
}

function Start-RecordedProcess([string]$Name, [string]$Executable, [string[]]$Arguments,
                               [string]$Directory, [double]$LimitSeconds) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.WorkingDirectory = $repository
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    $stdout = [IO.File]::Open((Join-Path $Directory "$Name.stdout.log"), [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $stderr = [IO.File]::Open((Join-Path $Directory "$Name.stderr.log"), [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $started = [DateTime]::UtcNow
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        $process = [Diagnostics.Process]::Start($info)
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $errTask = $process.StandardError.BaseStream.CopyToAsync($stderr)
    } catch {
        $stdout.Dispose()
        $stderr.Dispose()
        throw
    }
    [pscustomobject]@{name=$Name; process=$process; watch=$watch; stdout=$stdout; stderr=$stderr;
        outTask=$outTask; errTask=$errTask; executable=$Executable; arguments=$Arguments;
        directory=$Directory; started_utc=$started.ToString('o'); limit_s=$LimitSeconds; finished=$false}
}

function Finish-RecordedProcess($Job, [bool]$TimedOut = $false, [bool]$Interrupted = $false) {
    if ($Job.finished) { return }
    if (!$Job.process.HasExited) {
        $Job.process.Kill($true)
        if (!$Job.process.WaitForExit(5000)) { throw "Could not stop $($Job.name)" }
    }
    $Job.watch.Stop()
    $Job.outTask.GetAwaiter().GetResult()
    $Job.errTask.GetAwaiter().GetResult()
    $Job.stdout.Dispose()
    $Job.stderr.Dispose()
    $receipt = [ordered]@{phase=$Job.name; started_utc=$Job.started_utc;
        wall_s=$Job.watch.Elapsed.TotalSeconds; limit_s=$Job.limit_s; exit=$Job.process.ExitCode;
        timeout=$TimedOut; interrupted=$Interrupted; executable=$Job.executable; arguments=$Job.arguments}
    Write-AtomicJson (Join-Path $Job.directory "$($Job.name).receipt.json") $receipt
    $Job.finished = $true
    Write-Host ('{0} wall={1:F3}s exit={2} timeout={3}' -f $Job.name, $receipt.wall_s, $receipt.exit, $TimedOut)
    return [pscustomobject]$receipt
}

function Wait-RecordedProcess($Job) {
    while (!$Job.process.HasExited -and $Job.watch.Elapsed.TotalSeconds -le $Job.limit_s) {
        $null = $Job.process.WaitForExit(500)
    }
    $receipt = Finish-RecordedProcess $Job (!$Job.process.HasExited)
    if ($receipt.timeout -or $receipt.exit -ne 0) { throw "The $($Job.name) stage failed. Its logs and receipt are preserved." }
    return $receipt
}

try {
    $session = [IO.Path]::GetFullPath($SessionDir)
    $build = [IO.Path]::GetFullPath($BuildDirectory)
    if ($Resume -and !(Test-Path -LiteralPath (Join-Path $session 'manifest.json'))) {
        throw 'Resume requires an existing complete session manifest.'
    }
    if (!$Resume -and (Test-Path -LiteralPath $session)) {
        throw 'A new session requires an unused output directory. Use -Resume for a saved session.'
    }
    if (!$Resume) { $null = New-Item -Path $session -ItemType Directory }
    $lease = [IO.File]::Open((Join-Path $session '.workflow.lock'), [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)

    $sources = @(
        Get-ChildItem -LiteralPath 'src' -Recurse -File | Where-Object Extension -in @('.h', '.cpp', '.cu') | Select-Object -ExpandProperty FullName
        Get-ChildItem -LiteralPath $PSScriptRoot -File | Select-Object -ExpandProperty FullName
        (Resolve-Path 'tools/pond-flexible-tail-check/habitat_config.h.in').Path
    )
    $sourceBindings = @(Get-RelativeBindings $sources)
    $fingerprintText = ($sourceBindings | ForEach-Object { $_.path + ' ' + $_.sha256 }) -join "`n"
    $fingerprintText += "`nVisual Studio 17 2022|x64|Release|sm_120"
    $fingerprint = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($fingerprintText)))
    $gait = (Resolve-Path 'flexible-tail-preview/live_data/fitted-gait.bin').Path
    $offset = (Resolve-Path 'flexible-tail-preview/live_data/yaw-offset-update-8.bin').Path
    $starts = (Resolve-Path 'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin').Path
    $heldout = (Resolve-Path 'flexible-tail-preview/live_data/straight-evaluation-starts.bin').Path
    $initial = (Resolve-Path 'assets/research-evidence/pond-flexible-tail-chooser-bank-v2/data/training/actor-final.bin').Path
    $frozen = @(Get-RelativeBindings @($gait, $offset, $starts, $heldout,
        $initial, 'src/physics/flexible_tail.h', 'flexible-tail-preview/habitat.json'))
    $expected = @{
        'flexible-tail-preview/live_data/fitted-gait.bin'='2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6'
        'flexible-tail-preview/live_data/yaw-offset-update-8.bin'='52BA55F110DD1227CBFEB1E47E6199E50DE74687E82E9DD651114CDA86FEC354'
        'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin'='75E6AA2504726082E84E286FD5B3ADF2B7B9D4F7D4797116EF8A682F72DEC025'
        'flexible-tail-preview/live_data/straight-evaluation-starts.bin'='4A081CCC3DEA8C7310191902CC094E50FA6C0B341606DC9DBC4E63DD581FFF39'
        'assets/research-evidence/pond-flexible-tail-chooser-bank-v2/data/training/actor-final.bin'='8472921262B35B8837730D1A85EE5F6D97E45166DF88877669CF48D16D9519AF'
        'src/physics/flexible_tail.h'='B6878791EC1A3CC68112866DE8C105390A43AC143A8EDCBA9EA2BF2430CCB226'
        'flexible-tail-preview/habitat.json'='554E9E0976F22CD6B473EAAB84559C7B9887358C4B620CEF4592D5FFFEC0C7DF'
    }
    foreach ($item in $frozen) {
        if ($item.sha256 -ne $expected[$item.path]) { throw "Protected input differs at $($item.path)" }
    }

    if ($Resume) {
        $manifest = Get-Content -LiteralPath (Join-Path $session 'manifest.json') -Raw | ConvertFrom-Json
        if ($manifest.schema -ne 1 -or $manifest.steering_mode -ne $SteeringMode -or
            $manifest.max_updates -ne $TargetUpdates -or $manifest.total_training_wall_budget_s -ne $WallSeconds -or
            $manifest.source_fingerprint -ne $fingerprint) { throw 'Resume configuration or source identity differs from the saved session.' }
        Assert-Bindings $manifest.sources
        Assert-Bindings $manifest.frozen_inputs
    } else {
        $manifest = [ordered]@{schema=1; created_utc=[DateTime]::UtcNow.ToString('o');
            worlds=1024; decisions_per_rollout=8; minibatch=1024; epochs=4;
            max_updates=$TargetUpdates; total_training_wall_budget_s=$WallSeconds;
            evaluation_interval=8; steering_mode=$SteeringMode; source_fingerprint=$fingerprint;
            import_kind='legacy actor weights only, one-time initialization';
            sources=$sourceBindings; frozen_inputs=$frozen}
        Write-AtomicJson (Join-Path $session 'manifest.json') $manifest
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'SESSION_PROTOCOL.md') -Destination (Join-Path $session 'protocol.md')
        foreach ($binding in $sourceBindings) {
            $destination = Join-Path (Join-Path $session 'source') $binding.path
            $null = New-Item -Path (Split-Path -Parent $destination) -ItemType Directory -Force
            Copy-Item -LiteralPath (Join-Path $repository $binding.path) -Destination $destination
        }
    }

    $null = New-Item -Path $build -ItemType Directory -Force
    $stampPath = Join-Path $build 'build-fingerprint.json'
    $trainExe = Join-Path $build 'Release/pond_flexible_tail_chooser_session.exe'
    $evalExe = Join-Path $build 'Release/pond_flexible_tail_chooser_eval.exe'
    $rebuild = !(Test-Path -LiteralPath $trainExe) -or !(Test-Path -LiteralPath $evalExe)
    if (!$rebuild) {
        $rebuild = !(Test-Path -LiteralPath $stampPath) -or
            (Get-Content -LiteralPath $stampPath -Raw | ConvertFrom-Json).fingerprint -ne $fingerprint
    }
    if ($rebuild) {
        $preparations = Join-Path $session 'preparations'
        $null = New-Item -Path $preparations -ItemType Directory -Force
        $preparationIndex = @(Get-ChildItem -LiteralPath $preparations -Directory).Count + 1
        $preparation = Join-Path $preparations ('build-{0:D4}' -f $preparationIndex)
        $null = New-Item -Path $preparation -ItemType Directory
        $cmake = (Get-Command cmake).Source
        $configure = Start-RecordedProcess 'configure' $cmake @('-S', $PSScriptRoot, '-B', $build,
            '-G', 'Visual Studio 17 2022', '-A', 'x64') $preparation 600
        $active += $configure
        $null = Wait-RecordedProcess $configure
        $compile = Start-RecordedProcess 'build' $cmake @('--build', $build, '--config', 'Release', '--target',
            'pond_flexible_tail_chooser_session', 'pond_flexible_tail_chooser_eval') $preparation 600
        $active += $compile
        $null = Wait-RecordedProcess $compile
        Write-AtomicJson $stampPath @{fingerprint=$fingerprint; binaries=@(Get-RelativeBindings @($trainExe, $evalExe))}
    }
    $executablesPath = Join-Path $session 'executables.json'
    if ($Resume) {
        Assert-Bindings (Get-Content -LiteralPath $executablesPath -Raw | ConvertFrom-Json)
        Assert-Bindings (Get-Content -LiteralPath (Join-Path $session 'prerequisite-bindings.json') -Raw | ConvertFrom-Json)
    } else {
        $verificationDir = (Resolve-Path -LiteralPath $ResumeVerificationDir).Path
        $verificationPath = Join-Path $verificationDir 'verification.json'
        $receiptPath = Join-Path $verificationDir 'receipt.json'
        $verification = Get-Content -LiteralPath $verificationPath -Raw | ConvertFrom-Json
        $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        if (!$verification.pass -or !$receipt.pass -or $receipt.timeout -or $receipt.exit -ne 0 -or
            $receipt.steering_mode -ne $SteeringMode -or
            $receipt.exe_sha256 -ne (Get-FileHash -LiteralPath $trainExe).Hash -or
            $receipt.verification_sha256 -ne (Get-FileHash -LiteralPath $verificationPath).Hash) {
            throw 'Full-session resume verification has not passed for this exact executable and interface.'
        }
        $verificationInputs = @($receipt.frozen_inputs)
        foreach ($path in @($gait, $offset, $starts, $initial)) {
            $relative = [IO.Path]::GetRelativePath($repository, $path).Replace([IO.Path]::DirectorySeparatorChar, [char]'/')
            $entry = @($verificationInputs | Where-Object path -eq $relative)
            if ($entry.Count -ne 1 -or $entry[0].sha256 -ne (Get-FileHash -LiteralPath $path).Hash) {
                throw "Resume verification did not bind the session input $relative"
            }
        }
        $steeringDir = (Resolve-Path -LiteralPath $SteeringEvidenceDir).Path
        $steeringSummary = Join-Path $steeringDir 'assessment/summary.json'
        $steeringBindings = Join-Path $steeringDir 'bindings.json'
        if (!(Test-Path -LiteralPath $steeringSummary) -or !(Test-Path -LiteralPath $steeringBindings)) {
            throw 'The matched physical steering diagnostic evidence is missing.'
        }
        $prerequisites = @(Get-RelativeBindings @($verificationPath, $receiptPath, $steeringSummary, $steeringBindings))
        Write-AtomicJson (Join-Path $session 'prerequisite-bindings.json') $prerequisites
        $copyRoot = Join-Path $session 'prerequisites'
        $null = New-Item -Path $copyRoot -ItemType Directory
        Copy-Item -LiteralPath $verificationDir -Destination (Join-Path $copyRoot 'resume-verification') -Recurse
        Copy-Item -LiteralPath $steeringDir -Destination (Join-Path $copyRoot 'steering-diagnostic') -Recurse
        Write-AtomicJson $executablesPath @(Get-RelativeBindings @($trainExe, $evalExe))
    }
    $invocations = Join-Path $session 'invocations'
    $null = New-Item -Path $invocations -ItemType Directory -Force
    $number = @(Get-ChildItem -LiteralPath $invocations -Directory).Count + 1
    $invocation = Join-Path $invocations ('run-{0:D4}' -f $number)
    $null = New-Item -Path $invocation -ItemType Directory
    $usedWall = 0.0
    foreach ($receipt in Get-ChildItem -LiteralPath $invocations -Filter 'training.receipt.json' -Recurse -File) {
        $usedWall += [double](Get-Content -LiteralPath $receipt.FullName -Raw | ConvertFrom-Json).wall_s
    }
    $remaining = [math]::Floor($WallSeconds - $usedWall)
    if ($remaining -lt 1) { throw 'The session has spent its registered training wall budget.' }
    $training = Join-Path $session 'training'
    $arguments = if ($Resume) {
        @('resume', $gait, $offset, $starts, $training, "$TargetUpdates", "$remaining", $SteeringMode)
    } else {
        @('new', $gait, $offset, $starts, $initial, $training, "$TargetUpdates", "$remaining", $SteeringMode)
    }
    Write-AtomicJson (Join-Path $invocation 'launch.json') @{resume=[bool]$Resume;
        remaining_training_budget_s=$remaining; binaries=@(Get-FileHash -LiteralPath @($trainExe, $evalExe));
        arguments=$arguments; steering_mode=$SteeringMode}
    $trainer = Start-RecordedProcess 'training' $trainExe $arguments $invocation ($remaining + 600)
    $active += $trainer
    $evalRoot = Join-Path $session 'evaluations'
    $null = New-Item -Path $evalRoot -ItemType Directory -Force
    $evaluation = $null
    $trainReceipt = $null
    $finalQueued = $false
    $pending = [Collections.Generic.SortedSet[int]]::new()
    $checked = [Collections.Generic.HashSet[int]]::new()
    foreach ($directory in Get-ChildItem -LiteralPath $evalRoot -Directory) {
        if ($directory.Name -match '^update-(\d+)$' -and (Test-Path -LiteralPath (Join-Path $directory.FullName 'evaluation.receipt.json'))) {
            $null = $checked.Add([int]$Matches[1])
        }
    }

    while (!$trainer.finished -or $null -ne $evaluation -or $pending.Count -gt 0 -or !$finalQueued) {
        foreach ($file in Get-ChildItem -LiteralPath $training -Filter 'actor-*.bin' -File -ErrorAction SilentlyContinue) {
            if ($file.BaseName -match '^actor-(\d+)$') {
                $update = [int]$Matches[1]
                if ($update -gt 0 -and $update % 8 -eq 0 -and !$checked.Contains($update)) { $null = $pending.Add($update) }
            }
        }
        if (!$trainer.finished -and ($trainer.process.HasExited -or $trainer.watch.Elapsed.TotalSeconds -gt $trainer.limit_s)) {
            $trainReceipt = Finish-RecordedProcess $trainer (!$trainer.process.HasExited)
            if ($trainReceipt.timeout -or $trainReceipt.exit -ne 0) { throw 'The sustained trainer stopped on a fault or its external deadline.' }
            $result = Get-Content -LiteralPath (Join-Path $training 'result.json') -Raw | ConvertFrom-Json
            $last = [int]$result.updates_completed
            if ($last -gt 0 -and !$checked.Contains($last)) { $null = $pending.Add($last) }
            $finalQueued = $true
        }
        if ($null -ne $evaluation -and ($evaluation.process.HasExited -or $evaluation.watch.Elapsed.TotalSeconds -gt $evaluation.limit_s)) {
            $receipt = Finish-RecordedProcess $evaluation (!$evaluation.process.HasExited)
            if ($receipt.timeout -or $receipt.exit -notin @(0, 1)) { throw 'The held-out evaluator stopped before scoring its registered suite.' }
            $assessment = Join-Path $evaluation.directory 'assessment'
            $summary = Get-Content -LiteralPath (Join-Path $assessment 'summary.json') -Raw | ConvertFrom-Json
            if ([bool]$summary.pass -ne ($receipt.exit -eq 0)) { throw 'Evaluator status and summary disagree.' }
            $cases = @(Import-Csv -LiteralPath (Join-Path $assessment 'cases.csv'))
            if (@($cases | Where-Object { [int]$_.fault -in @(1, 3, 6) }).Count) { throw 'A held-out trajectory reported invalid numerical or runtime state.' }
            $candidate = Get-Content -LiteralPath (Join-Path $evaluation.directory 'candidate.json') -Raw | ConvertFrom-Json
            if ((Get-FileHash -LiteralPath $candidate.path).Hash -ne $candidate.sha256) { throw 'An exported candidate changed during evaluation.' }
            Write-AtomicJson (Join-Path $evaluation.directory 'admission.json') @{qualified=[bool]$summary.pass;
                actor=$candidate; summary=$summary; godot_published=$false}
            $evaluation = $null
        }
        if ($null -eq $evaluation -and $pending.Count -gt 0) {
            $update = $pending.Min
            $null = $pending.Remove($update)
            $null = $checked.Add($update)
            $directory = Join-Path $evalRoot ('update-{0:D4}' -f $update)
            if (Test-Path -LiteralPath $directory) { throw "Incomplete prior evaluation found at $directory" }
            $null = New-Item -Path $directory -ItemType Directory
            $actor = Join-Path $training "actor-$update.bin"
            Write-AtomicJson (Join-Path $directory 'candidate.json') @{update=$update; path=$actor;
                sha256=(Get-FileHash -LiteralPath $actor).Hash; steering_mode=$SteeringMode}
            $evalArguments = @($gait, $offset, $heldout, $actor, (Join-Path $directory 'assessment'), $SteeringMode)
            $evaluation = Start-RecordedProcess 'evaluation' $evalExe $evalArguments $directory 20
            $active += $evaluation
            Write-Host "Evaluating immutable chooser update $update while the training session retains its full state."
        }
        if (!$trainer.finished -or $null -ne $evaluation) { Start-Sleep -Milliseconds 250 }
    }
    Assert-Bindings $sourceBindings
    Assert-Bindings $frozen
    Write-AtomicJson (Join-Path $invocation 'completion.json') @{completed_utc=[DateTime]::UtcNow.ToString('o');
        training=$trainReceipt; evaluated_updates=@($checked | Sort-Object); godot_published=$false}
} finally {
    foreach ($job in $active) {
        if (!$job.finished) { $null = Finish-RecordedProcess $job $false $true }
        $job.process.Dispose()
    }
    if ($lease) { $lease.Dispose() }
    Pop-Location
}
