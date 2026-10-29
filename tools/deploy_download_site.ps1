param(
    [string]$ResourceGroup='rg-tadpole-pond',
    [string]$AppName='tadpole-pond'
)
$ErrorActionPreference='Stop'
$workspace=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$site=Join-Path $workspace 'website/dist'
$cli=Join-Path $workspace 'build/site-tools/node_modules/@azure/static-web-apps-cli/dist/cli/bin.js'
if(-not (Test-Path -LiteralPath $cli)) {throw 'Install the local deployment tool: npm install --prefix build/site-tools @azure/static-web-apps-cli --no-audit --no-fund'}
$sku=az staticwebapp show --name $AppName --resource-group $ResourceGroup --query sku.name --output tsv
if($LASTEXITCODE -ne 0 -or $sku.Trim() -ne 'Free') {throw 'Expected the approved Free static website; refusing a different hosting SKU'}
$previousToken=$env:SWA_CLI_DEPLOYMENT_TOKEN
try {
    # Credential stays local and is sent only to the authorized Azure website.
    # Never put it in an argument, file, log, Git remote or GitHub secret.
    $env:SWA_CLI_DEPLOYMENT_TOKEN=az staticwebapp secrets list --name $AppName --resource-group $ResourceGroup --query properties.apiKey --output tsv
    if($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($env:SWA_CLI_DEPLOYMENT_TOKEN)) {throw 'Azure deployment credential unavailable'}
    & node $cli deploy $site --env production --no-use-keychain
    if($LASTEXITCODE -ne 0) {throw 'Static website deployment failed'}
} finally {
    $env:SWA_CLI_DEPLOYMENT_TOKEN=$previousToken
    $previousToken=$null
}
