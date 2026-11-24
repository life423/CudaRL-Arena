#Requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Protocol,
    [Parameter(Mandatory)][string]$ProtocolSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function ReadJson($path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
function Digest($path){(Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant()}
function SaveJson($path,$value){[IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 12))}
function Need($ok,$why){if(!$ok){throw $why}}
$planFile=(Resolve-Path -LiteralPath $Protocol).Path
$plan=ReadJson $planFile
Need ((Digest $planFile) -eq $ProtocolSha256 -and $plan.protocol_id -in @('pond-motor-residual-ppo-v1-2026-11-24','pond-motor-residual-four-ppo-v1-2026-11-24')) 'Sealed residual protocol required'
foreach($name in @('trainer','evaluator','parent_policy','cpu_gate','cuda_gate','mirror_result','critic_source_candidate','critic_source_protocol')){
    Need ((Digest $plan.($name+'_path_absolute')) -eq $plan.($name+'_sha256')) "Sealed input changed: $name"
}
Need ((ReadJson $plan.mirror_result_path_absolute).physical_symmetry_passed) 'Physics reflection failed'
$root=Split-Path $plan.ledger_path_absolute
$logs=Join-Path $root 'runner'
Need (!(Test-Path -LiteralPath $logs)) 'This allocation has already been attempted'
New-Item -ItemType Directory -Path $logs|Out-Null
$script:Stages=[Collections.Generic.List[object]]::new()
$watch=[Diagnostics.Stopwatch]::StartNew()
SaveJson (Join-Path $logs 'started.json') @{protocol_sha256=$ProtocolSha256;started_utc=[DateTimeOffset]::UtcNow.ToString('o')}
function Stage([string]$name,[string]$exe,[string[]]$arguments){
    $timer=[Diagnostics.Stopwatch]::StartNew()
    & $exe @arguments 1> (Join-Path $logs "$name.stdout.txt") 2> (Join-Path $logs "$name.stderr.txt")
    $code=$LASTEXITCODE
    $script:Stages.Add(@{stage=$name;executable=$exe;arguments=$arguments;seconds=$timer.Elapsed.TotalSeconds;exit_code=$code})
    Need ($code -eq 0) "Stage $name stopped. Its original output is preserved."
}
function CheckSkills($assessment){
    Need ($assessment.assessment_completed -and $assessment.retention_passed -and !$assessment.hard_failure) 'Required physics or skill retention failed'
    Need ($assessment.turn_cases_attempted -eq 16 -and $assessment.short_swim_cases_passed -eq 4 -and $assessment.rest_cases_passed -eq 14 -and $assessment.retained_parent_long_window_passes -eq 28) 'Incomplete required assessment'
}
$state='STOPPED'
$lastUpdate=0
$reason=''
try {
    $initial=Join-Path $root 'init'
    Stage 'initialize' $plan.trainer_path_absolute @('--initialize-only','--protocol',$planFile,'--out',$initial)
    $candidate=Join-Path $initial 'residual-0.bin'
    $baseline=Join-Path $root 'baseline0'
    Stage 'baseline0' $plan.evaluator_path_absolute @('--run','--baseline','--reuse-verified-baseline','--parent',$plan.parent_policy_path_absolute,'--residual-candidate',$candidate,'--protocol',$planFile,'--out',$baseline,'--ledger',$plan.evaluation_ledger_path_absolute,'--long-reference',$plan.long_reference_path_absolute)
    CheckSkills (ReadJson (Join-Path $baseline 'result.json'))
    $resume=Join-Path $initial 'state-0.bin'
    $receipt=Join-Path $baseline 'baseline-receipt.bin'
    foreach($update in @(8,16,24,32)){
        $cycle=[Diagnostics.Stopwatch]::StartNew()
        $chunk=Join-Path $root "updates-$($update-7)-$update"
        Stage "train-$update" $plan.trainer_path_absolute @('--run','--protocol',$planFile,'--out',$chunk,'--resume',$resume,'--retention-receipt',$receipt)
        $candidate=Join-Path $chunk "residual-$update.bin"
        $resume=Join-Path $chunk "state-$update.bin"
        $evaluation=Join-Path $root "eval$update"
        Stage "eval$update" $plan.evaluator_path_absolute @('--run','--turn-first','--parent',$plan.parent_policy_path_absolute,'--residual-candidate',$candidate,'--protocol',$planFile,'--out',$evaluation,'--ledger',$plan.evaluation_ledger_path_absolute,'--long-reference',$plan.long_reference_path_absolute)
        $assessment=ReadJson (Join-Path $evaluation 'result.json')
        CheckSkills $assessment
        $lastUpdate=$update
        $receipt=Join-Path $evaluation 'retention-receipt.bin'
        Need ((Digest $plan.parent_policy_path_absolute) -eq $plan.parent_policy_sha256 -and (Digest $planFile) -eq $ProtocolSha256) 'Protected actor or protocol changed'
        $result=@{update=$update;turns_passed=$assessment.turn_cases_passed;turns_attempted=$assessment.turn_cases_attempted;long_windows=$assessment.retained_parent_long_window_passes;rests=$assessment.rest_cases_passed;short_swims=$assessment.short_swim_cases_passed;cycle_seconds=$cycle.Elapsed.TotalSeconds}
        SaveJson (Join-Path $root "cycle-$update.json") $result
        $result|ConvertTo-Json -Compress
        if($assessment.turning_qualified){$state='REQUESTED_TURNS_QUALIFIED';break}
    }
    if($state -ne 'REQUESTED_TURNS_QUALIFIED'){$state='COMPLETED_ALLOCATION'}
}catch{$reason=$_.Exception.Message;throw}
finally{
    SaveJson (Join-Path $logs 'result.json') @{status=$state;last_verified_update=$lastUpdate;reason=$reason;elapsed_seconds=$watch.Elapsed.TotalSeconds;protocol_sha256=$ProtocolSha256;stages=$script:Stages.ToArray();original_motor_sha256=(Digest $plan.parent_policy_path_absolute)}
}
