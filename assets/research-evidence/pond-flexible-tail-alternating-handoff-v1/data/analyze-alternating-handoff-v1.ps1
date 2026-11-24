
$ErrorActionPreference='Stop'
$study='build/flexible-tail/alternating-handoff-v1/assessment'
$prior='build/flexible-tail/beat-timing-v1/assessment'
$archivedEvents=@(Import-Csv "$prior/beat-events.csv" | Where-Object {$_.axis -eq 'root' -and $_.policy -eq 'zero'})
$archivedSegments=@(Import-Csv "$prior/segments.csv" | Where-Object policy -eq 'zero')

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

$gates=@(foreach($sequence in 'left_start','right_start'){
    $dir=Join-Path $study $sequence
    $result=Get-Content "$dir/result.json" -Raw|ConvertFrom-Json
    $roots=@(Import-Csv "$dir/beat-events.csv"|Where-Object axis -eq 'root')
    $segments=@(Import-Csv "$dir/segments.csv")
    $decisions=@(Import-Csv "$dir/decisions.csv")
    $side=if($sequence -eq 'left_start'){'left'}else{'right'}
    $sign=if($side -eq 'left'){1}else{-1}
    $case='moving_a_'+$side
    $stage2=[int]$result.stage2_fork_control
    $stage3=[int]$result.stage3_fork_control
    if($stage2 -lt 700 -or $stage2 -gt 800 -or $stage3-$stage2 -lt 700 -or $stage3-$stage2 -gt 800){
        throw "Unexpected cue times in $sequence"
    }
    foreach($branch in 'prefix','stage2_control','stage2_test','stage3_control','stage3_test'){
        $commands=@($decisions|Where-Object branch -eq $branch)
        if(!$commands.Count -or @($commands|Where-Object offset_enabled -ne '1').Count){
            throw "Offset activation changed in $sequence $branch"
        }
    }

    $prefix=@($roots|Where-Object branch -eq 'prefix')
    $stage2Path=@($roots|Where-Object {$_.branch -eq 'prefix' -or $_.branch -eq 'stage2_test'})
    $throughStage3=@($stage2Path|Where-Object {[int]$_.confirmed_sample -le $stage3*16})
    foreach($turn in 1,2,3){
        $requestedSign=if($turn -eq 2){-$sign}else{$sign}
        foreach($window in 'late_comparison','onward'){
            if($turn -eq 1){
                $label=if($window -eq 'onward'){$side+'_onward'}else{$side+'_turn'}
                $archivedLabel=if($window -eq 'onward'){'onward'}else{'late_turn'}
                $actual=$segments|Where-Object {$_.branch -eq 'prefix' -and $_.segment -eq $label}
                $reference=$archivedSegments|Where-Object {$_.case -eq $case -and $_.segment -eq $archivedLabel}
                $refEvents=@($archivedEvents|Where-Object case -eq $case)
                $trialEvents=$stage2Path
            }else{
                $name='stage'+$turn
                $actual=$segments|Where-Object {$_.branch -eq ($name+'_test') -and $_.segment -eq $window}
                $reference=$segments|Where-Object {$_.branch -eq ($name+'_control') -and $_.segment -eq $window}
                $history=if($turn -eq 2){@($prefix)}else{@($throughStage3)}
                $refEvents=@($history)+@($roots|Where-Object branch -eq ($name+'_control'))
                $trialEvents=@($history)+@($roots|Where-Object branch -eq ($name+'_test'))
            }
            if(!$actual -or !$reference){throw "Missing segment $sequence turn $turn $window"}
            $lo=[double]$actual.window_start_s
            $hi=[double]$actual.window_end_s
            if($hi-$lo -lt 1.99 -or $hi-$lo -gt 2.01){throw 'Unexpected measurement window'}
            $refPeriods=@(Get-Beats $refEvents $lo $hi)
            $periods=@(Get-Beats $trialEvents $lo $hi)
            if(!$refPeriods.Count -or !$periods.Count){throw "Complete beat evidence unavailable for $sequence $turn $window"}
            $refPeriod=$refPeriods | Measure-Object -Average
            $testPeriod=$periods | Measure-Object -Average -Maximum
            $frequencyRatio=$refPeriod.Average/$testPeriod.Maximum
            $rootRatio=[double]$actual.root_swing_rad/[double]$reference.root_swing_rad
            $passiveRatio=[double]$actual.passive_swing_rad/[double]$reference.passive_swing_rad
            [pscustomobject]@{sequence=$sequence;turn=$turn;window=$window;start_s=$lo;end_s=$hi;
                signed_course_deg=$actual.signed_course_change_deg;course_error_deg=$actual.course_error_deg;
                projected_speed_mm_s=$actual.projected_speed_mm_s;matched_error_deg=$reference.course_error_deg;
                test_mean_period_s=$testPeriod.Average;matched_mean_period_s=$refPeriod.Average;
                min_frequency_ratio=$frequencyRatio;root_swing_ratio=$rootRatio;passive_swing_ratio=$passiveRatio;
                reversals=$actual.root_reversals;matched_reversals=$reference.root_reversals;
                pass=($actual.motion_pass -eq '1' -and $requestedSign*[double]$actual.signed_course_change_deg -gt 0 -and
                      $frequencyRatio -ge .8 -and $rootRatio -ge .8 -and $passiveRatio -ge .8)}
        }
    }
})
$gates|Export-Csv "$study/alternating-gates.csv" -NoTypeInformation
$gates|ConvertTo-Csv -NoTypeInformation

