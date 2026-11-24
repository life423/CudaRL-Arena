#Requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Protocol,
    [Parameter(Mandatory)][string]$ProtocolSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$watch=[Diagnostics.Stopwatch]::StartNew()
function ReadJson($path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
function Digest($path){(Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant()}
function SaveJson($path,$value){[IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 10))}
function Need($ok,$message){if(!$ok){throw $message}}
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$planPath=(Resolve-Path -LiteralPath $Protocol).Path
Need ((Digest $planPath) -eq $ProtocolSha256) 'Protocol hash changed'
$plan=ReadJson $planPath
Need ($plan.protocol_id -eq 'pond-motor-mirror-distill-v1-2026-11-24') 'Wrong experiment'
foreach($name in @('trainer','evaluator','mirror_result','parent_candidate','candidate_source_protocol','human_protocol')){
    Need ((Digest $plan.($name+'_path_absolute')) -eq $plan.($name+'_sha256')) "Sealed input changed: $name"
}
Need ((ReadJson $plan.mirror_result_path_absolute).physical_symmetry_passed) 'Physical mirror did not pass'
$root=Split-Path $plan.output_path_absolute
$runner=Join-Path $root 'runner'
Need (!(Test-Path -LiteralPath $runner)) 'This experiment was already attempted'
New-Item -ItemType Directory -Path $runner|Out-Null
$times=[Collections.Generic.List[object]]::new()
SaveJson (Join-Path $runner 'started.json') @{protocol_sha256=$ProtocolSha256;started_utc=[DateTimeOffset]::UtcNow.ToString('o')}
$success=$false
try {
    $phase=[Diagnostics.Stopwatch]::StartNew()
    & $plan.trainer_path_absolute --run --protocol $planPath --out $plan.output_path_absolute 1> (Join-Path $runner 'training.stdout.txt') 2> (Join-Path $runner 'training.stderr.txt')
    $times.Add(@{stage='supervised_training';seconds=$phase.Elapsed.TotalSeconds;exit_code=$LASTEXITCODE})
    Need ($LASTEXITCODE -eq 0) 'Supervised training stopped, see retained evidence'
    $actor=Join-Path $plan.output_path_absolute 'distilled-actor.bin'
    Need (Test-Path -LiteralPath $actor) 'Missing trained actor'
    $evaluation=Join-Path $root 'evaluation'
    $phase.Restart()
    & $plan.evaluator_path_absolute --run --parent $plan.parent_policy_path_absolute --candidate $plan.parent_candidate_path_absolute --distilled-actor $actor --protocol $planPath --out $evaluation --ledger (Join-Path $root 'evaluation-ledger.bin') --long-reference $plan.long_reference_path_absolute 1> (Join-Path $runner 'evaluation.stdout.txt') 2> (Join-Path $runner 'evaluation.stderr.txt')
    $times.Add(@{stage='physical_evaluation';seconds=$phase.Elapsed.TotalSeconds;exit_code=$LASTEXITCODE})
    $assessment=ReadJson (Join-Path $evaluation 'result.json')
    $success=$LASTEXITCODE -eq 0 -and $assessment.retention_passed -and !$assessment.hard_failure
    Need ((Digest $planPath) -eq $ProtocolSha256 -and (Digest $plan.parent_candidate_path_absolute) -eq $plan.parent_candidate_sha256 -and (Digest $plan.parent_policy_path_absolute) -eq $plan.parent_policy_sha256) 'Protected inputs changed'
    $assessment|ConvertTo-Json -Depth 5
} finally {
    SaveJson (Join-Path $runner 'result.json') @{protocol_sha256=$ProtocolSha256;completed=$success;stages=$times.ToArray();through_physical_assessment_seconds=$watch.Elapsed.TotalSeconds;new_PPO_updates=0;distilled_actor_adopted=$false}
}
