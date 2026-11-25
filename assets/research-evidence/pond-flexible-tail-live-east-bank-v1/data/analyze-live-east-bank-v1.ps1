$ErrorActionPreference='Stop'
$study='build/flexible-tail/live-east-bank-v1'
$ref='assets/research-evidence/pond-flexible-tail-alternating-handoff-v1/data'
$bank=Join-Path $study 'assessment'
$prior=@(Import-Csv "$ref/alternating-gates.csv")
$results=Get-Content "$bank/result.json" -Raw|ConvertFrom-Json
function Get-Beats($rows,[double]$begin,[double]$end){
    $events=@($rows | Sort-Object {[int]$_.confirmed_sample} -Unique)
    for($i=0;$i -lt $events.Count-2;$i++){
        $first=$events[$i];$middle=$events[$i+1];$last=$events[$i+2]
        if($first.direction -ne $last.direction -or $first.direction -eq $middle.direction){
            throw 'Nonalternating root reversal events in a branch'
        }
        if(!$first.crossing_s -or !$last.crossing_s){throw 'Missing root crossing time'}
        $a=[double]$first.crossing_s; $b=[double]$last.crossing_s
        $mid=($a+$b)/2
        if($mid -ge $begin -and $mid -lt $end){$b-$a}
    }
}


$gates=@(foreach($side in 'left_start','right_start'){
    $result=Get-Content "$bank/$side/result.json" -Raw|ConvertFrom-Json
    $source=Import-Csv "$ref/$side/forks.csv"|Where-Object stage -eq 'stage2'
    if($result.source_digest -ne $source.checksum -or $result.cue_control -ne $source.control){
        throw "The live cue does not match the archived state for $side"
    }
    $offsetActive=@(Import-Csv "$bank/$side/decisions.csv"|Where-Object branch -eq 'east_bank')
    if(!$offsetActive.Count -or @($offsetActive|Where-Object offset_enabled -ne '1').Count){
        throw "The frozen yaw offset was inactive during $side"
    }
    $root=@(Import-Csv "$bank/$side/beat-events.csv"|Where-Object axis -eq 'root')
    $archive=@(Import-Csv "$ref/$side/segments.csv"|Where-Object branch -eq 'stage2_control')
    foreach($window in Import-Csv "$bank/$side/segments.csv"){
        $w=$window.segment
        $bounds=$prior|Where-Object {$_.sequence -eq $side -and $_.turn -eq '2' -and $_.window -eq $w}
        $hold=$archive|Where-Object segment -eq $w
        $beats=@(Get-Beats $root ([double]$window.window_start_s) ([double]$window.window_end_s))
        if(!$beats.Count -or !$bounds -or !$hold){throw "Incomplete matched beat analysis for $side $w"}
        $frequency=[double]$bounds.matched_mean_period_s / ($beats|Measure-Object -Maximum).Maximum
        $rootSwing=[double]$window.root_swing_rad / [double]$hold.root_swing_rad
        $passiveSwing=[double]$window.passive_swing_rad / [double]$hold.passive_swing_rad
        [pscustomobject]@{sequence=$side;turn_deg=$result.turn_deg;window=$w;setback_mm=$result.setback_mm;
            signed_course_deg=$window.signed_course_change_deg;course_error_deg=$window.course_error_deg;
            projected_speed_mm_s=$window.projected_speed_mm_s;clear_water_error_deg=$bounds.course_error_deg;
            frequency_ratio=$frequency;root_swing_ratio=$rootSwing;passive_swing_ratio=$passiveSwing;
            root_reversals=$window.root_reversals;matched_root_reversals=$hold.root_reversals;
            pass=($result.motion_pass -and $window.motion_pass -eq '1' -and $frequency -ge .8 -and $rootSwing -ge .8 -and $passiveSwing -ge .8)}
    }
})
$gates|Export-Csv "$bank/live-bank-gates.csv" -NoTypeInformation
$gates|ConvertTo-Csv -NoTypeInformation

