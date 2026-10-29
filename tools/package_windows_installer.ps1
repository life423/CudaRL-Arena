param(
    [Parameter(Mandatory=$true)][string]$MakensisPath,
    [string]$PackageResult='build/windows-release/package-result.json',
    [switch]$AllowDirty
)
$ErrorActionPreference='Stop'
$workspace=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$package=Get-Content -LiteralPath $PackageResult -Raw | ConvertFrom-Json
$manifest=Get-Content -LiteralPath (Join-Path $package.output 'release-manifest.json') -Raw | ConvertFrom-Json
if(($manifest.sourceDirty -or [bool](& git -C $workspace status --porcelain)) -and -not $AllowDirty) {
    throw 'Public installers require a clean source and clean payload. -AllowDirty is for local verification only.'
}
if($manifest.version -ne '0.1.0-preview.2') {throw 'Repackage the app as 0.1.0-preview.2 before building its installer.'}
if(-not $AllowDirty -and $manifest.sourceCommit -ne (& git -C $workspace rev-parse HEAD).Trim()) {throw 'Repackage from the current clean commit first.'}
$expected=@($manifest.files.path)+@('release-manifest.json')
$actual=@(Get-ChildItem -LiteralPath $package.output -Recurse -File | ForEach-Object {$_.FullName.Substring($package.output.Length+1).Replace('\','/')})
if(Compare-Object ($expected | Sort-Object) ($actual | Sort-Object)) {throw 'Payload contains missing or extra files; create a fresh package.'}
foreach($file in $manifest.files) {
    $path=Join-Path $package.output $file.path
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw "Payload hash mismatch: $($file.path)"
    }
}
$output=Join-Path $workspace 'build/windows-release/TadpolePondSetup.exe'
$icon=Join-Path $workspace 'tools/windows/TadpolePond.ico'
& $MakensisPath /V2 "/DPAYLOAD=$($package.output)" "/DOUTPUT=$output" "/DAPPICON=$icon" (Join-Path $PSScriptRoot 'windows/tadpole-pond.nsi')
if($LASTEXITCODE -ne 0) {throw 'NSIS compilation failed.'}
$hash=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
@("$hash  TadpolePondSetup.exe", "$($package.sha256)  TadpolePond-Windows-x64.zip") | Set-Content -LiteralPath (Join-Path $workspace 'build/windows-release/SHA256SUMS.txt') -Encoding ascii
[ordered]@{installer=$output;bytes=(Get-Item -LiteralPath $output).Length;sha256=$hash;version=$manifest.version;sourceCommit=$manifest.sourceCommit;sourceDirty=$manifest.sourceDirty} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workspace 'build/windows-release/installer-result.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $workspace 'build/windows-release/installer-result.json')
