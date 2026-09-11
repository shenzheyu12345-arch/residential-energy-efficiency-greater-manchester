param(
    [string]$Python = "python",
    [switch]$SkipBoundaryDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$arguments = @("run_pipeline.py")
if ($SkipBoundaryDownload) {
    $arguments += "--skip-boundary-download"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Pipeline failed with exit code $LASTEXITCODE"
}

