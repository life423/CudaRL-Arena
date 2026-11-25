#Requires -Version 7.0
param([Parameter(Mandatory)][switch]$Run, [switch]$Posture)

$ErrorActionPreference = 'Stop'
$trialClock = [Diagnostics.Stopwatch]::StartNew()
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = Join-Path $repo 'build/turn-course-rate-v1'
$trainer = Join-Path $repo 'build/collection-replay-v3/native/four/Release/pond_residual_ppo.exe'
$evaluator = Join-Path $repo 'build/turn-course-rate-evaluator/native/Release/pond_candidate_check.exe'
$priorPath = Join-Path $repo 'build/turn39-residual-four/protocol.json'
$dynamics = Join-Path $repo 'assets/research-evidence/pond-broadphase-dynamics-v2/result.json'
$replay = Join-Path $repo 'assets/research-evidence/pond-residual-four-collection-replay-v1/result.json'
if ($Posture) {
    $root = Join-Path $repo 'build/turn-posture-v1'
    $trainer = Join-Path $repo 'build/collection-replay-v3/native/four/Release/pond_turn_ppo_posture_v1.exe'
    $evaluator = Join-Path $repo 'build/turn-posture-evaluator/native/Release/pond_candidate_check.exe'
    $replay = Join-Path $repo 'assets/research-evidence/pond-residual-four-posture-replay-v1/result.json'
}
function Digest([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Json([string]$Path) { Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable }
function WriteJson([string]$Path, $Value) { [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 12)) }
function Need($Condition, [string]$Reason) { if (!$Condition) { throw $Reason } }

$trainerInfo = & $trainer --describe --classified-clearance | ConvertFrom-Json -AsHashtable
Need ($LASTEXITCODE -eq 0 -and $trainerInfo.max_updates -eq 8) 'Incorrect trainer allocation'
$evaluatorInfo = & $evaluator --describe | ConvertFrom-Json -AsHashtable
Need ($LASTEXITCODE -eq 0 -and $evaluatorInfo.turn_cases -eq 16) 'Incorrect turn evaluator'
$replayReceipt = Json $replay
Need ($replayReceipt.passed -and $replayReceipt.physical_controls -eq 32768 -and $replayReceipt.bindings.source_sha256 -eq $trainerInfo.source_sha256) 'The compiled four-output collector has not qualified'
$prior = Json $priorPath
$original = $prior.parent_policy_sha256
$results = [Collections.Generic.List[object]]::new()
$armNames = @('reference', 'course_rate')
if ($Posture) {
    $armNames = @('posture')
    $referenceProtocol = Join-Path $repo 'build/turn-course-rate-v1/comparison/reference/protocol.json'
    $referenceTraining = Join-Path $repo 'build/turn-course-rate-v1/comparison/reference/updates-1-8/result.json'
    $referenceAssessment = Join-Path $repo 'build/turn-course-rate-v1/comparison/reference/assessment/result.json'
    $referenceCases = Join-Path $repo 'build/turn-course-rate-v1/comparison/reference/assessment/cases.csv'
    $reference = Json $referenceProtocol
    $referenceResult = Json $referenceTraining
    Need ($reference.reward_arm -eq 'reference' -and $referenceResult.passed -and $referenceResult.completed_updates -eq 8 -and
          $reference.parent_policy_sha256 -eq $original -and (Json $referenceAssessment).turn_cases_attempted -eq 16) 'Missing matched original-reward result'
}
$protocols = @{}
$created = New-Item -ItemType Directory -Path (Join-Path $root 'comparison')
$evidence = $created.FullName
$firstUpdate = Join-Path $evidence 'first-confirmed-optimizer.json'
$stageLog = Join-Path $evidence 'stages.json'

function Stage([string]$Arm, [string]$Name, [string]$Exe, [string[]]$CliArgs, [string]$ProgressCsv = '') {
    $stageLimitMs = 4000
    if ($Posture) {
        $stageLimitMs = [Math]::Min($stageLimitMs, 20000 - $trialClock.ElapsedMilliseconds)
        Need ($stageLimitMs -gt 0) 'The 20-second trial limit was reached before starting the next stage'
    }
    $processInfo = [Diagnostics.ProcessStartInfo]::new($Exe)
    $processInfo.WorkingDirectory = $repo
    $processInfo.CreateNoWindow = $true
    $processInfo.UseShellExecute = $false
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    foreach ($arg in $CliArgs) { $processInfo.ArgumentList.Add($arg) }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $p = [Diagnostics.Process]::Start($processInfo)
    $stdout = $p.StandardOutput.ReadToEndAsync()
    $stderr = $p.StandardError.ReadToEndAsync()
    while (!$p.HasExited -and $clock.Elapsed.TotalMilliseconds -lt $stageLimitMs) {
        if ($ProgressCsv -and !(Test-Path -LiteralPath $firstUpdate) -and (Test-Path -LiteralPath $ProgressCsv)) {
            $progress = Get-Content -LiteralPath $ProgressCsv -ErrorAction SilentlyContinue | Select-Object -Skip 1 -First 1
            if ($progress -match '^1,32768,327680,32,32,') {
                WriteJson $firstUpdate @{arm=$Arm;updated=1;actor_Adam_calls=32;critic_Adam_calls=32;utc=[DateTimeOffset]::UtcNow.ToString('o')}
            }
        }
        $remaining = [Math]::Max(0, [int]($stageLimitMs - $clock.ElapsedMilliseconds))
        $null = $p.WaitForExit([Math]::Min(35, $remaining))
    }
    $timedOut = !$p.HasExited
    if ($timedOut) { $p.Kill(); $p.WaitForExit() }
    $clock.Stop()
    $log = Join-Path $evidence "$Arm-$Name"
    [IO.File]::WriteAllText("$log.stdout.txt", $stdout.GetAwaiter().GetResult())
    [IO.File]::WriteAllText("$log.stderr.txt", $stderr.GetAwaiter().GetResult())
    $code = $p.ExitCode
    $p.Dispose()
    $results.Add(@{arm=$Arm;stage=$Name;ms=$clock.Elapsed.TotalMilliseconds;exit_code=$code;watchdog_executed=$timedOut})
    WriteJson $stageLog $results.ToArray()
    Need (!$timedOut -and $code -eq 0 -and $clock.Elapsed.TotalMilliseconds -lt $stageLimitMs) "Bounded stage $Arm/$Name stopped. Its logs and ledger are preserved."
}

foreach ($arm in $armNames) {
    $path = (New-Item -ItemType Directory -Path (Join-Path $evidence $arm)).FullName
    $plan = [ordered]@{
        protocol_id = $trainerInfo.protocol_id
        reward_arm = $arm
        max_updates = 8
        worlds = 512
        rollout_controls = 64
        actor_learning_rate = 0.0003
        critic_learning_rate = 0.001
        epochs = 4
        minibatch = 4096
        max_process_wall_seconds = 4
        max_training_controls = 262144
        max_training_substeps = 2621440
        max_actor_Adam_calls = 256
        max_critic_Adam_calls = 256
        max_assessment_controls = 36800
        source_sha256 = $trainerInfo.source_sha256
        qualified_physics_source_sha256 = $prior.qualified_physics_source_sha256
        classified_physics_source_sha256 = $trainerInfo.qualified_physics_source_sha256
        retention_evaluator_source_fnv1a64 = $evaluatorInfo.eval_source_identity_fnv1a64
        ledger_path_absolute = Join-Path $path 'training-ledger.bin'
        evaluation_ledger_path_absolute = Join-Path $path 'evaluation-ledger.bin'
        trainer_sha256 = Digest $trainer
        evaluator_sha256 = Digest $evaluator
        classified_dynamics_path_absolute = $dynamics
        classified_dynamics_sha256 = Digest $dynamics
        classified_collection_path_absolute = $replay
        classified_collection_sha256 = Digest $replay
        request_schedule = 'Mirrored ±15, ±30, ±60 and ±90 degrees, stratified from the first request. Original 8 mm/s, settling, hold and timeout settings.'
        acceptance = 'Score 16 existing turns and the eight ±60/90 cases separately. Eight of eight large cases establishes their direct-turn milestone. Retention gates unchanged.'
        reward = $(if ($arm -eq 'course_rate') { 'Multiply positive swimming reward during issued turns by .5+.5*exp(-((actual_half_second_course_rate-clamp(signed_course_error/2s, +/-10deg/s))/7.5deg/s)^2); quality .5 with invalid course below 3mm/s.' } else { 'Existing turn reward.' })
    }
    if ($Posture) {
        $plan.reward = 'During accepted, nonterminal active turns subtract .025*(1-exp(-(max(0, tilt_deg-5)/5)^2)) per control from the original reward.'
        $plan.max_trial_wall_seconds = 20
        $plan.reference_protocol_sha256 = Digest $referenceProtocol
        $plan.reference_training_sha256 = Digest $referenceTraining
        $plan.reference_assessment_sha256 = Digest $referenceAssessment
        $plan.reference_cases_sha256 = Digest $referenceCases
        foreach ($key in @('worlds','rollout_controls','actor_learning_rate','critic_learning_rate','epochs','minibatch','request_schedule')) {
            Need ($plan[$key] -eq $reference[$key]) "Reference and posture allocation differ at $key"
        }
    }
    foreach ($key in @('parent_policy','cpu_gate','cuda_gate','long_reference','critic_source_candidate','critic_source_protocol',
                       'previous_protocol','previous_candidate_zero','previous_baseline_receipt','previous_baseline_result')) {
        $inputPath = $prior[$key+'_path_absolute']
        $expected = $prior[$key+'_sha256']
        Need ((Digest $inputPath) -eq $expected) "Changed sealed input $key"
        $plan[$key+'_path_absolute'] = $inputPath
        $plan[$key+'_sha256'] = $expected
    }
    $protocol = Join-Path $path 'protocol.json'
    WriteJson $protocol $plan
    $protocols[$arm] = @{plan=$plan;path=$path;protocol=$protocol;digest=(Digest $protocol)}
}

foreach ($arm in $armNames) {
    $entry = $protocols[$arm]
    $plan = $entry.plan
    $initial = Join-Path $entry.path 'init'
    $baseline = Join-Path $entry.path 'baseline'
    $chunk = Join-Path $entry.path 'updates-1-8'
    $assessment = Join-Path $entry.path 'assessment'
    $evalArgs = @('--run','--protocol',$entry.protocol,'--parent',$plan.parent_policy_path_absolute,
        '--ledger',$plan.evaluation_ledger_path_absolute,'--long-reference',$plan.long_reference_path_absolute)
    Need ((Digest $entry.protocol) -eq $entry.digest -and (Digest $trainer) -eq $plan.trainer_sha256 -and (Digest $evaluator) -eq $plan.evaluator_sha256) 'Sealed stage changed'
    Stage $arm 'init' $trainer @('--initialize-only','--classified-clearance','--protocol',$entry.protocol,'--out',$initial)
    Stage $arm 'baseline' $evaluator ($evalArgs + @('--baseline','--reuse-verified-baseline','--residual-candidate',(Join-Path $initial 'residual-0.bin'),'--out',$baseline))
    $before = Json (Join-Path $baseline 'result.json')
    Need ($before.retention_passed -and $before.turn_cases_attempted -eq 16) 'Baseline receipt failed'
    Stage $arm 'train' $trainer @('--run','--classified-clearance','--protocol',$entry.protocol,'--out',$chunk,
        '--resume',(Join-Path $initial 'state-0.bin'),'--retention-receipt',(Join-Path $baseline 'baseline-receipt.bin')) (Join-Path $chunk 'updates.csv')
    $trained = Json (Join-Path $chunk 'result.json')
    Need ($trained.passed -and $trained.completed_updates -eq 8 -and $trained.base_Adam_calls -eq 0) 'Training checkpoint incomplete'
    if (!(Test-Path -LiteralPath $firstUpdate)) {
        WriteJson $firstUpdate @{arm=$arm;updated=$trained.completed_updates;actor_Adam_calls=$trained.actor_Adam_calls;critic_Adam_calls=$trained.critic_Adam_calls}
    }
    Stage $arm 'fixed-turns' $evaluator ($evalArgs + @('--turn-first','--residual-candidate',(Join-Path $chunk 'residual-8.bin'),'--out',$assessment))
    $after = Json (Join-Path $assessment 'result.json')
    Need ($after.assessment_completed -and $after.retention_passed -and !$after.hard_failure -and $after.turn_cases_attempted -eq 16) 'Assessment did not pass required gates'
    Need ((Digest $plan.parent_policy_path_absolute) -eq $original) 'Original motor changed'
}
$allocation = if ($Posture) { 'one eight-update arm against the archived original-reward reference' } else { 'two arms, 8 updates each' }
WriteJson (Join-Path $evidence 'completed.json') @{allocation=$allocation;original_motor_sha256=$original;wall_ms=$trialClock.Elapsed.TotalMilliseconds;stages=$results.ToArray()}
