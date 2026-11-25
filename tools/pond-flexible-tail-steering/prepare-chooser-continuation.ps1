param(
    [string]$SourceTraining='build/flexible-tail/seven-options-v6-bank-cap16/stage-06-bank/training',
    [Parameter(Mandatory=$true)][string]$DestinationTraining
)
$ErrorActionPreference='Stop'
Add-Type -Path (Join-Path $PSScriptRoot 'BudgetCopy.cs')
$source=(Resolve-Path -LiteralPath $SourceTraining).Path
$receiptPath=Join-Path (Split-Path -Parent $source) 'training.receipt.json'
$receipt=Get-Content -LiteralPath $receiptPath -Raw|ConvertFrom-Json
if($receipt.exit -ne -1 -or $receipt.timeout -or [double]$receipt.wall_s -le 0){throw 'Expected preserved externally interrupted first invocation receipt.'}
$destination=[IO.Path]::GetFullPath($DestinationTraining)
$result=[BankResumeBudgetCopy]::Prepare($source,$destination,[double]$receipt.wall_s)
$report=[ordered]@{
    schema=1;pass=$true;physics_steps=0;optimizer_updates=0;
    source_training=$source;destination_training=$destination;source_receipt=$receiptPath;
    source_receipt_sha256=(Get-FileHash -LiteralPath $receiptPath).Hash;
    exact_checkpoint0_preserved=$true;exact_actor0_preserved=$true;source_files_unchanged=$true;
    old_target_updates=$result.OldTarget;new_target_updates=$result.NewTarget;
    original_used_wall_s=$result.OldUsed;charged_interrupted_wall_s=[double]$receipt.wall_s;
    copied_used_wall_s=$result.NewUsed;wall_limit_s=$result.Limit;
    checkpoint_payload_checksum=$result.CheckpointHash.ToString();
    reserved_unobserved_updates=1;remaining_completed_updates_maximum=63;
    unobserved_actions_maximum=8192;unobserved_optimizer_steps_maximum_each=32;
    unobserved_physics_substeps_maximum=10485760;
    changed_files=@('budget.checkpoint');
    copied_files=@('latest.checkpoint','checkpoints/session-0.bin','actor-0.bin');
    command=@('resume','GAIT','MOTOR','STARTS',$destination,'63','1200','0')
}
$json=$report|ConvertTo-Json -Depth 8
$reportPath=Join-Path $destination 'continuation-budget.json'
[IO.File]::WriteAllText($reportPath,$json,[Text.UTF8Encoding]::new($false))
$json
