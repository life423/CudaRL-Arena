param([Parameter(Mandatory=$true)][string]$Parent,[Parameter(Mandatory=$true)][string]$Child,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
function Read-Starts([string]$Path){
    $bytes=[IO.File]::ReadAllBytes($Path)
    if([BitConverter]::ToUInt64($bytes,0) -ne [uint64]0x31504B434C524443 -or [BitConverter]::ToUInt64($bytes,8) -ne ($bytes.Length-24)){throw 'Checkpoint container mismatch'}
    # Counts remains the exact 72-byte legacy structure; its array is the last
    # complete payload field in FTSS4 and FTSS5. No body decoding or stepping.
    $begin=$bytes.Length-1024*72-8
    if([BitConverter]::ToUInt64($bytes,$begin) -ne 1024){throw 'Unexpected final Counts array'}
    $result=@();for($i=0;$i -lt 1024;$i++){$at=$begin+8+$i*72;$result+=[pscustomobject]@{lane=$i;starts=[BitConverter]::ToUInt32($bytes,$at+40);left=[BitConverter]::ToUInt32($bytes,$at+56);right=[BitConverter]::ToUInt32($bytes,$at+60)}}
    return $result
}
$parentCounts=@(Read-Starts $Parent);$childCounts=@(Read-Starts $Child)
$counts=@{left15=0L;right15=0L;left30=0L;right30=0L};$rows=@()
for($i=0;$i -lt 1024;$i++){
    # This parent is the first, non-forked V4 study: its per-lane start count
    # equals the persistent episode/RNG index. Child imports that index intact.
    $first=[int]$parentCounts[$i].starts;$starts=[int]$childCounts[$i].starts
    $direction=if(($i%2) -eq 0){'left'}else{'right'};$fifteen=0;$thirty=0
    for($j=0;$j -lt $starts;$j++){$angle=if((([int][Math]::Floor($i/2)+$first+$j) -band 1) -eq 1){30}else{15};$counts[($direction+$angle)]++;if($angle -eq 15){$fifteen++}else{$thirty++}}
    if(($direction -eq 'left' -and $childCounts[$i].left -ne $starts) -or ($direction -eq 'right' -and $childCounts[$i].right -ne $starts)){throw 'Sign/start counters disagree'}
    $rows+=[pscustomobject]@{lane=$i;imported_episode_index=$first;new_starts=$starts;new15=$fifteen;new30=$thirty;direction=$direction}
}
$rows|Export-Csv ($Output+'.csv') -NoTypeInformation
[ordered]@{schema=1;physics_steps=0;optimizer_updates=0;method='Decode existing terminal Counts arrays; original V4 per-lane starts equal imported episode index; apply unchanged mixed index/episode parity';parent_sha256=(Get-FileHash $Parent).Hash;child_sha256=(Get-FileHash $Child).Hash;starts=$counts;total_new_starts=($childCounts.starts|Measure-Object -Sum).Sum;imported_parent_starts=($parentCounts.starts|Measure-Object -Sum).Sum;initial_inherited_30_episodes_excluded_from_new_starts=$true}|ConvertTo-Json -Depth 5|Set-Content ($Output+'.json')
Get-Content ($Output+'.json')
