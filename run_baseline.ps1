# AI-assisted: OpenAI Codex (GPT-6), OpenAI, 2026-10-23.
param(
    [Parameter(Mandatory=$true)][string]$Source,
    [string]$Audit = (Join-Path $PSScriptRoot 'outputs/p0_audit_20261023_final'),
    [string]$Output = (Join-Path $PSScriptRoot ('outputs/baseline_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))),
    [string]$Config = (Join-Path $PSScriptRoot 'configs/baseline_v1.json'),
    [string]$Python = (Join-Path $PSScriptRoot '.venv/Scripts/python.exe')
)
$ErrorActionPreference = 'Stop'
$priorPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    & $Python -m beer_models.baseline --source $Source --audit $Audit --config $Config --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Baseline failed with exit code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $priorPythonPath
}
