# AI-assisted: OpenAI Codex (GPT-6), OpenAI, 2026-10-23.
param(
    [Parameter(Mandatory=$true)][string]$Source,
    [string]$Output = (Join-Path $PSScriptRoot ('outputs/audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))),
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$priorPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    & $Python -m beer_audit --source $Source --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Audit failed with exit code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $priorPythonPath
}
