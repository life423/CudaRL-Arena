param(
    [string]$Original = 'build/pond-reliability-study/preview-original/replay.json',
    [string]$Trained = 'build/pond-reliability-study/preview-trained/replay.json',
    [string]$Output = 'build/pond-reliability-study/bank-preview/replay.json'
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw 'Choose a new preview output file; existing recordings are preserved.' }
$originalData = Get-Content -LiteralPath $Original -Raw | ConvertFrom-Json
$trainedData = Get-Content -LiteralPath $Trained -Raw | ConvertFrom-Json
foreach ($recording in @($originalData, $trainedData)) {
    if ($recording.version -ne 1 -or $recording.cases.Count -ne 1 -or $recording.cases[0].frames.Count -ne 1201) {
        throw 'Expected a native version-1, one-case, sixty-second recording.'
    }
}
$first = $originalData.cases[0]
$second = $trainedData.cases[0]
foreach ($field in @('position', 'orientation', 'foods', 'links', 'angles')) {
    $before = ConvertTo-Json -InputObject $first.frames[0].$field -Depth 20 -Compress
    $after = ConvertTo-Json -InputObject $second.frames[0].$field -Depth 20 -Compress
    if ($before -ne $after) { throw ('Recording starts differ: ' + $field) }
}
if ($first.motor_update -ne $second.motor_update -or $first.current_cm_s -ne $second.current_cm_s) {
    throw 'Swimming policy and current must match.'
}
$combined = [ordered]@{
    version = 1
    source = 'Native CUDA held-out bank replay; frozen selectors, same motor and physics; no live training.'
    cases = @($first, $second)
}
$absoluteOutput = [IO.Path]::GetFullPath($Output)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($absoluteOutput)) | Out-Null
[IO.File]::WriteAllText($absoluteOutput, (ConvertTo-Json -InputObject $combined -Depth 100 -Compress), [Text.UTF8Encoding]::new($false))
Write-Output ('Saved paired recovery preview: ' + $absoluteOutput)
