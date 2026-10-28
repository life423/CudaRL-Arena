$ErrorActionPreference='Stop'
$baseline='docs/experiments/swimmer/gentle-currents/frozen-policy.bin'
$pool='build/pond-bank-study/expanded-0xBAA7D001/pool.bin'
$fixture='docs/experiments/swimmer/gentle-currents/diagnosis/stall-at-60s.bin'
$original=Import-Csv build/pond-bank-study/dev-original/summary.csv
$originalRate=[double]$original.recovered_20s/[double]$original.trials
$records=@()
foreach ($key in @(71,109)) {
    $learning='build/pond-bank-study/seed-'+$key+'/learning.csv'
    $lines=Get-Content -LiteralPath $learning
    if ($lines[0].StartsWith('update,')) {$normal=Import-Csv $learning}
    else {$normal=Import-Csv $learning -Header update,calm_fpm,px_current_fpm,diagonal_current_fpm,wall_seconds}
    foreach ($update in @(128,192,256,320,384)) {
        $candidate='build/pond-bank-study/seed-'+$key+'/candidate-'+$update+'.bin'
        if (!(Test-Path -LiteralPath $candidate)) {throw ('Missing completed checkpoint '+$candidate)}
        $output='build/pond-bank-study/development/seed-'+$key+'-update-'+$update
        & build/Release/pond_bank_study.exe --policy $baseline --pool $pool --progress $candidate --seconds 60 --output $output *> ($output+'.log')
        if ($LASTEXITCODE -ne 0) {throw ('Bank development evaluation failed '+$candidate)}
        & build/Release/pond_current_eval.exe --progress $candidate --fixture $fixture --output ($output+'/fixture') --trials 64 --seconds 60 *> ($output+'-fixture.log')
        if ($LASTEXITCODE -ne 0) {throw ('Fixture evaluation failed '+$candidate)}
        $bank=Import-Csv ($output+'/summary.csv')
        $known=Import-Csv ($output+'/fixture/fixture.csv') | Where-Object mode -eq candidate
        $score=$normal | Where-Object {[int]$_.update -eq $update} | Select-Object -Last 1
        $rate=[double]$bank.recovered_20s/[double]$bank.trials
        $calm=[double]$score.calm_fpm
        $current=([double]$score.px_current_fpm+[double]$score.diagonal_current_fpm)/2
        $eligible=($rate -ge $originalRate+.1 -and $calm -ge 17.765625*.98 -and $current -ge ((14.640625+13.96875)/2)*.98 -and [double]$known.first_food_upper_seconds -gt 0 -and [double]$known.first_food_upper_seconds -le 20)
        $records+=[pscustomobject]@{Seed=$key;Update=$update;PhaseUpdates=$update-128;Recovered20=[int]$bank.recovered_20s;Trials=[int]$bank.trials;RecoveryRate=$rate;BankMeanFoods=[double]$bank.mean_foods;CalmFpm=$calm;CurrentMeanFpm=$current;FixtureFirst=[double]$known.first_food_upper_seconds;Eligible=$eligible;Path=$candidate}
        Write-Output ('Development seed={0} update={1} recovered={2}/{3} eligible={4}' -f $key,$update,$bank.recovered_20s,$bank.trials,$eligible)
    }
}
$records | Export-Csv build/pond-bank-study/selection.csv -NoTypeInformation
$winner=$records | Where-Object Eligible | Sort-Object @{Expression='RecoveryRate';Descending=$true},@{Expression='BankMeanFoods';Descending=$true},PhaseUpdates,Seed | Select-Object -First 1
if (!$winner) {throw 'No checkpoint passes the predeclared development gate'}
$winner | ConvertTo-Json | Set-Content build/pond-bank-study/selected.json
Copy-Item -LiteralPath $winner.Path -Destination docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin
$winner | ConvertTo-Json
