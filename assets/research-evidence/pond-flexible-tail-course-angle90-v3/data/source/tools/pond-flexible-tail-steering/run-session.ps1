param(
    [string]$SessionDir = 'build/flexible-tail/fast-steering-v1',
    [string]$BuildDirectory = 'build/flexible-tail/fast-steering-v1-build',
    [ValidateSet('half-second','whole-beat','whole-beat-90')][string]$FeedbackMode = 'half-second',
    [switch]$Resume,
    [ValidateRange(1,128)][int]$TargetUpdates = 128,
    [ValidateRange(60,1200)][int]$WallSeconds = 1200
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$jobs = [Collections.Generic.List[object]]::new()
$lease = $null
Push-Location -LiteralPath $repository

function Write-Json([string]$Path, $Value) {
    $Value | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath ($Path+'.tmp') -Encoding utf8
    [IO.File]::Move($Path+'.tmp', $Path, $true)
}
function Bind-Files([string[]]$Paths) {
    foreach ($path in ($Paths | Sort-Object -Unique)) {
        $absolute = (Resolve-Path -LiteralPath $path).Path
        [pscustomobject]@{path=[IO.Path]::GetRelativePath($repository,$absolute).Replace('\','/');
            sha256=(Get-FileHash -LiteralPath $absolute).Hash}
    }
}
function Assert-Bindings($Bindings) {
    foreach ($binding in $Bindings) {
        $path = Join-Path $repository $binding.path
        if (!(Test-Path -LiteralPath $path) -or (Get-FileHash -LiteralPath $path).Hash -ne $binding.sha256) {
            throw "Bound input changed: $($binding.path)"
        }
    }
}
function Start-JobProcess([string]$Name,[string]$Exe,[string[]]$Arguments,[string]$Directory,[double]$Limit) {
    $null = New-Item -Path $Directory -ItemType Directory -Force
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName=$Exe; $info.WorkingDirectory=$repository
    $info.UseShellExecute=$false; $info.CreateNoWindow=$true
    $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    $stdout=[IO.File]::Open((Join-Path $Directory "$Name.stdout.log"),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $stderr=[IO.File]::Open((Join-Path $Directory "$Name.stderr.log"),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {
        $started=[DateTime]::UtcNow; $watch=[Diagnostics.Stopwatch]::StartNew()
        $process=[Diagnostics.Process]::Start($info)
        $outTask=$process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $errTask=$process.StandardError.BaseStream.CopyToAsync($stderr)
    } catch { $stdout.Dispose();$stderr.Dispose();throw }
    $job=[pscustomobject]@{name=$Name;process=$process;watch=$watch;stdout=$stdout;stderr=$stderr;
        outTask=$outTask;errTask=$errTask;exe=$Exe;arguments=$Arguments;directory=$Directory;
        limit=$Limit;started_utc=$started.ToString('o');finished=$false;receipt=$null}
    $jobs.Add($job)
    return $job
}
function Finish-JobProcess($Job,[bool]$Timeout=$false,[bool]$Interrupted=$false) {
    if ($Job.finished) { return $Job.receipt }
    if (!$Job.process.HasExited) { $Job.process.Kill($true);$null=$Job.process.WaitForExit(5000) }
    $Job.watch.Stop()
    $Job.outTask.GetAwaiter().GetResult();$Job.errTask.GetAwaiter().GetResult()
    $Job.stdout.Dispose();$Job.stderr.Dispose()
    $receipt=[pscustomobject]@{phase=$Job.name;started_utc=$Job.started_utc;
        wall_s=$Job.watch.Elapsed.TotalSeconds;limit_s=$Job.limit;exit=$Job.process.ExitCode;
        timeout=$Timeout;interrupted=$Interrupted;executable=$Job.exe;arguments=$Job.arguments}
    Write-Json (Join-Path $Job.directory "$($Job.name).receipt.json") $receipt
    $Job.finished=$true;$Job.receipt=$receipt
    Write-Host ('{0}: {1:F3}s, exit {2}, timeout {3}' -f $Job.name,$receipt.wall_s,$receipt.exit,$Timeout)
    return $receipt
}
function Wait-JobProcess($Job) {
    while (!$Job.process.HasExited -and $Job.watch.Elapsed.TotalSeconds -lt $Job.limit) {
        $null=$Job.process.WaitForExit(250)
    }
    $receipt=Finish-JobProcess $Job (!$Job.process.HasExited)
    if ($receipt.exit -ne 0 -or $receipt.timeout) { throw "$($Job.name) failed; receipt and outputs retained." }
    return $receipt
}

try {
    $session=[IO.Path]::GetFullPath($SessionDir);$build=[IO.Path]::GetFullPath($BuildDirectory)
    if (!$Resume -and (Test-Path -LiteralPath $session)) { throw 'New session requires an unused directory.' }
    if ($Resume -and !(Test-Path -LiteralPath (Join-Path $session 'manifest.json'))) { throw 'Resume manifest missing.' }
    if (!$Resume) { $null=New-Item -Path $session -ItemType Directory }
    $lease=[IO.File]::Open((Join-Path $session '.workflow.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $runWatch=[Diagnostics.Stopwatch]::StartNew()
    $gait=(Resolve-Path 'flexible-tail-preview/live_data/fitted-gait.bin').Path
    $originalOffset=(Resolve-Path 'flexible-tail-preview/live_data/yaw-offset-update-8.bin').Path
    $offset=if ($FeedbackMode -eq 'whole-beat-90') {
        (Resolve-Path 'build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin').Path
    } else { $originalOffset }
    $starts=(Resolve-Path 'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin').Path
    $turns=(Resolve-Path 'build/flexible-tail/yaw-beat-v1/capture/eval-turn-worlds.bin').Path
    $straight=(Resolve-Path 'flexible-tail-preview/live_data/straight-evaluation-starts.bin').Path
    $inputs=@(Bind-Files @($gait,$offset,$originalOffset,$starts,$turns,$straight,'src/physics/flexible_tail.h','flexible-tail-preview/habitat.json'))
    $expected=@{
        'flexible-tail-preview/live_data/fitted-gait.bin'='2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6'
        'flexible-tail-preview/live_data/yaw-offset-update-8.bin'='52BA55F110DD1227CBFEB1E47E6199E50DE74687E82E9DD651114CDA86FEC354'
        'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin'='75E6AA2504726082E84E286FD5B3ADF2B7B9D4F7D4797116EF8A682F72DEC025'
        'build/flexible-tail/yaw-beat-v1/capture/eval-turn-worlds.bin'='4902A5050A09101D8E7115EE43DDF99F3288B99EF539A525548A5D3FC727BF1E'
        'flexible-tail-preview/live_data/straight-evaluation-starts.bin'='4A081CCC3DEA8C7310191902CC094E50FA6C0B341606DC9DBC4E63DD581FFF39'
        'src/physics/flexible_tail.h'='B6878791EC1A3CC68112866DE8C105390A43AC143A8EDCBA9EA2BF2430CCB226'
        'flexible-tail-preview/habitat.json'='554E9E0976F22CD6B473EAAB84559C7B9887358C4B620CEF4592D5FFFEC0C7DF'
        'build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin'='888BA2A571CEB9F90091D147A4CC56C6CECD092B3B6F58D679FBFCC779E98891'
    }
    foreach ($binding in $inputs) { if ($binding.sha256 -ne $expected[$binding.path]) { throw "Protected input changed: $($binding.path)" } }
    $sourcePaths=@(
        Get-ChildItem -LiteralPath 'src' -File -Recurse | Where-Object Extension -in @('.h','.cpp','.cu') | Select-Object -ExpandProperty FullName
        Get-ChildItem -LiteralPath $PSScriptRoot -File | Select-Object -ExpandProperty FullName
        (Resolve-Path 'tools/pond-flexible-tail-check/habitat_config.h.in').Path
    )
    $sources=@(Bind-Files $sourcePaths)
    $fingerprintText=($sources | ForEach-Object {$_.path+' '+$_.sha256}) -join "`n"
    $fingerprint=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($fingerprintText)))
    if ($Resume) {
        $manifest=Get-Content -LiteralPath (Join-Path $session 'manifest.json') -Raw | ConvertFrom-Json
        if ($manifest.source_fingerprint -ne $fingerprint -or $manifest.max_updates -ne $TargetUpdates -or
            $manifest.training_wall_budget_s -ne $WallSeconds -or $manifest.feedback -ne $FeedbackMode) { throw 'Resume source, observer or budget mismatch.' }
        Assert-Bindings $manifest.sources;Assert-Bindings $manifest.inputs
    } else {
        Write-Json (Join-Path $session 'manifest.json') @{schema=1;created_utc=[DateTime]::UtcNow.ToString('o');
            source_fingerprint=$fingerprint;sources=$sources;inputs=$inputs;worlds=1024;decisions=64;
            max_updates=$TargetUpdates;training_wall_budget_s=$WallSeconds;evaluation_interval=8;feedback=$FeedbackMode}
        $protocolName=switch ($FeedbackMode) {
            'whole-beat-90' {'COURSE_ANGLE90_PROTOCOL.md'}
            'whole-beat' {'BEAT_FEEDBACK_PROTOCOL.md'}
            default {'PROTOCOL.md'}
        }
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $protocolName) -Destination (Join-Path $session 'protocol.md')
        foreach ($binding in $sources) {
            $destination=Join-Path (Join-Path $session 'source') $binding.path
            $null=New-Item -Path (Split-Path -Parent $destination) -ItemType Directory -Force
            Copy-Item -LiteralPath (Join-Path $repository $binding.path) -Destination $destination
        }
    }
    $trainExe=Join-Path $build 'Release/pond_flexible_tail_steering_ppo.exe'
    $evalExe=Join-Path $build 'Release/pond_flexible_tail_steering_evaluate.exe'
    $feedbackCheckExe=Join-Path $build 'Release/pond_flexible_tail_beat_feedback_check.exe'
    $stamp=Join-Path $build 'fingerprint.json'
    $rebuild=!(Test-Path -LiteralPath $stamp) -or !(Test-Path -LiteralPath $trainExe) -or !(Test-Path -LiteralPath $evalExe)
    if (!$rebuild) { $rebuild=(Get-Content -LiteralPath $stamp -Raw | ConvertFrom-Json).source_fingerprint -ne $fingerprint }
    if ($Resume -and $rebuild) { throw 'Resume requires original bound binaries; rebuild as a new study.' }
    if ($rebuild) {
        $cmake=(Get-Command cmake).Source;$prep=Join-Path $session 'compilation'
        $null=Wait-JobProcess (Start-JobProcess 'configure' $cmake @('-S',$PSScriptRoot,'-B',$build,'-G','Visual Studio 17 2022','-A','x64') $prep 600)
        $null=Wait-JobProcess (Start-JobProcess 'compile' $cmake @('--build',$build,'--config','Release','--parallel','4') $prep 600)
        Write-Json $stamp @{source_fingerprint=$fingerprint}
    }
    if ($Resume) { Assert-Bindings (Get-Content -LiteralPath (Join-Path $session 'executables.json') -Raw | ConvertFrom-Json) }
    else { Write-Json (Join-Path $session 'executables.json') @(Bind-Files @($trainExe,$evalExe,$feedbackCheckExe)) }
    if (!$Resume) {
      if ($FeedbackMode -in @('whole-beat','whole-beat-90')) {
        $equivalenceCopy=Join-Path $session 'equivalence'
        $checkArguments=@($gait,$offset,$starts,$turns,$straight,(Join-Path $equivalenceCopy 'data'))
        if ($FeedbackMode -eq 'whole-beat-90') { $checkArguments+=@($FeedbackMode) }
        $null=Wait-JobProcess (Start-JobProcess 'feedback-check' $feedbackCheckExe $checkArguments $equivalenceCopy 30)
        $equivalence=Get-Content -LiteralPath (Join-Path $equivalenceCopy 'data/report.json') -Raw | ConvertFrom-Json
        if (!$equivalence.pass -or $equivalence.physics_steps -ne 0 -or $equivalence.optimizer_calls -ne 0) {
            throw 'Whole-beat feedback prerequisite failed before physical execution.'
        }
      } else {
        $equivalenceRoot=(Resolve-Path 'build/fast-task-diagnostic').Path
        $equivalence=Get-Content -LiteralPath (Join-Path $equivalenceRoot 'report.json') -Raw | ConvertFrom-Json
        $equivalenceBindings=Get-Content -LiteralPath (Join-Path $equivalenceRoot 'bindings.json') -Raw | ConvertFrom-Json
        foreach ($binding in $equivalenceBindings) {
            if ((Get-FileHash -LiteralPath $binding.path).Hash -ne $binding.sha256) { throw 'Offline equivalence evidence is stale.' }
        }
        foreach ($check in @('sparse_gait_vs_full_nn','four_weight_mean_vs_full_nn','historical_observer','critic_steering','inactive_bypass')) {
            if ($equivalence.$check.bit_differences -ne 0) { throw 'Frozen-controller equivalence prerequisite failed.' }
        }
        if ($equivalence.physics_steps -ne 0 -or $equivalence.optimizer_calls -ne 0 -or
            $equivalence.actual_seed_equal_state_pairs -ne 512 -or $equivalence.unchanged_bodies -ne 2048) {
            throw 'Invalid offline equivalence prerequisite.'
        }
        $equivalenceCopy=Join-Path $session 'equivalence'
        $null=New-Item -Path $equivalenceCopy -ItemType Directory
        foreach ($file in @('report.json','bindings.json','requests.csv','check.cpp','CMakeLists.txt')) {
            Copy-Item -LiteralPath (Join-Path $equivalenceRoot $file) -Destination (Join-Path $equivalenceCopy $file)
        }
      }
        $baseline=Join-Path $session 'baseline'
        $initialCandidate=if ($FeedbackMode -eq 'whole-beat-90') {
            Join-Path $equivalenceCopy 'data/initial-actor90.bin'
        } else { $offset }
        $null=Wait-JobProcess (Start-JobProcess 'baseline' $evalExe @($gait,$offset,$initialCandidate,$turns,$straight,(Join-Path $baseline 'assessment'),$FeedbackMode) $baseline 30)
        $baselineResult=Get-Content -LiteralPath (Join-Path $baseline 'assessment/result.json') -Raw | ConvertFrom-Json
        if ($baselineResult.global_faults -ne 0 -or $baselineResult.candidate_straight_retention -ne 12 -or
            !$baselineResult.all_inactive_commands_exact -or !$baselineResult.paired_retention_final_states_exact) {
            throw 'Baseline native or swimming-retention prerequisite failed; training did not start.'
        }
        $verification=Join-Path $session 'verification'
        $null=Wait-JobProcess (Start-JobProcess 'verification' $trainExe @('--verify-resume',$gait,$offset,$starts,(Join-Path $verification 'data'),'590',$FeedbackMode) $verification 600)
        $verificationResult=Get-Content -LiteralPath (Join-Path $verification 'data/verification.json') -Raw | ConvertFrom-Json
        if (!$verificationResult.pass -or $verificationResult.physical_updates_executed -ne 3) { throw 'Exact full CUDA resume prerequisite failed.' }
        if ($FeedbackMode -eq 'whole-beat-90') {
            if ((Get-FileHash -LiteralPath $initialCandidate).Hash -ne
                (Get-FileHash -LiteralPath (Join-Path $verification 'data/actor-0.bin')).Hash) {
                throw 'The tested transferred actor differs from the native trainer initialization.'
            }
        }
    }
    $invocations=Join-Path $session 'invocations'
    $null=New-Item -Path $invocations -ItemType Directory -Force
    $index=@(Get-ChildItem -LiteralPath $invocations -Directory).Count+1
    $invocation=Join-Path $invocations ('run-{0:D4}' -f $index)
    $usedWall=0.0
    foreach ($receipt in Get-ChildItem -LiteralPath $invocations -Filter 'training.receipt.json' -File -Recurse) { $usedWall+=(Get-Content -LiteralPath $receipt.FullName -Raw | ConvertFrom-Json).wall_s }
    $remaining=[math]::Floor($WallSeconds-$usedWall)
    if ($remaining -lt 1) { throw 'Registered training wall budget spent.' }
    $training=Join-Path $session 'training'
    $mode=if ($Resume) {'--resume'} else {'--new'}
    $job=Start-JobProcess 'training' $trainExe @($mode,$gait,$offset,$starts,$training,"$TargetUpdates","$WallSeconds",$FeedbackMode) $invocation ($remaining+600)
    $scheduled=[Collections.Generic.HashSet[int]]::new()
    $evaluations=[Collections.Generic.Queue[object]]::new()
    $evalJob=$null
    while (!$job.finished -or $evalJob -or $evaluations.Count -gt 0) {
        for ($update=8;$update -le $TargetUpdates;$update+=8) {
            $actor=Join-Path $training "actor-$update.bin"
            $directory=Join-Path (Join-Path $session 'evaluations') ('update-{0:D3}' -f $update)
            if ((Test-Path -LiteralPath $actor) -and !$scheduled.Contains($update)) {
                $null=$scheduled.Add($update)
                if (!(Test-Path -LiteralPath (Join-Path $directory 'evaluation.receipt.json'))) { $evaluations.Enqueue(@{update=$update;actor=$actor;directory=$directory}) }
            }
        }
        if (!$evalJob -and $evaluations.Count -gt 0) {
            $next=$evaluations.Dequeue()
            $evalJob=Start-JobProcess 'evaluation' $evalExe @($gait,$offset,$next.actor,$turns,$straight,(Join-Path $next.directory 'assessment'),$FeedbackMode) $next.directory 30
        }
        if ($evalJob -and ($evalJob.process.HasExited -or $evalJob.watch.Elapsed.TotalSeconds -gt $evalJob.limit)) {
            $receipt=Finish-JobProcess $evalJob (!$evalJob.process.HasExited)
            if ($receipt.exit -ne 0 -or $receipt.timeout) {
                $null=New-Item -Path (Join-Path $training 'STOP') -ItemType File -Force
                throw 'Evaluation process failed; cooperative training stop requested.'
            }
            $summaryPath=Join-Path $evalJob.directory 'assessment/result.json'
            $assessment=Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            Write-Host (Get-Content -LiteralPath $summaryPath -Raw)
            if ($assessment.global_faults -ne 0 -or $assessment.candidate_straight_retention -ne 12 -or
                !$assessment.all_inactive_commands_exact -or !$assessment.paired_retention_final_states_exact) {
                $null=New-Item -Path (Join-Path $training 'STOP') -ItemType File -Force
                $null=Wait-JobProcess $job
                throw 'Native or skill-retention gate failed; training stopped at a complete update.'
            }
            $evalJob=$null
        }
        if (!$job.finished -and ($job.process.HasExited -or $job.watch.Elapsed.TotalSeconds -gt $job.limit)) {
            $receipt=Finish-JobProcess $job (!$job.process.HasExited)
            if ($receipt.exit -ne 0 -or $receipt.timeout) { throw 'Training failed; no automatic restart.' }
        }
        if (!$job.finished -or $evalJob -or $evaluations.Count -gt 0) { [Threading.Thread]::Sleep(100) }
    }
    Assert-Bindings $inputs
    $runWatch.Stop()
    Write-Json (Join-Path $invocation 'workflow.json') @{pass=$true;wall_s=$runWatch.Elapsed.TotalSeconds;
        evaluated_updates=@($scheduled | Sort-Object);compilation_separate=$true;original_inputs_unchanged=$true}
} catch {
    foreach ($job in $jobs) { if (!$job.finished) { $null=Finish-JobProcess $job $false $true } }
    throw
} finally {
    if ($lease) { $lease.Dispose() }
    Pop-Location
}
