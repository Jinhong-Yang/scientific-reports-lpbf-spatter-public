param(
    [int]$W09ProcessId = 0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-w05\Scripts\python.exe'
$logRoot = Join-Path $projectRoot 'runs\new_study\pretest_pipeline'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Invoke-Stage {
    param([string]$Name, [string[]]$Arguments)
    $started = Get-Date
    Write-Output "START $Name $($started.ToString('o'))"
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $ended = Get-Date
    Write-Output "COMPLETE $Name $($ended.ToString('o'))"
}

Set-Location -LiteralPath $projectRoot
if ($W09ProcessId -gt 0) {
    $process = Get-Process -Id $W09ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-Output "WAIT W09 PID=$W09ProcessId"
        Wait-Process -Id $W09ProcessId
    }
}

$w09 = Get-Content -LiteralPath (Join-Path $projectRoot 'evidence\generator_factorial\W09_COMPLETENESS.json') -Raw | ConvertFrom-Json
if ($w09.status -ne 'COMPLETE' -or $w09.completed -ne 40 -or $w09.failed_or_interrupted -ne 0) {
    throw 'W09 did not reach immutable 40/40 completion'
}

Invoke-Stage -Name 'W09_ANALYSIS' -Arguments @('src\analysis\analyze_w09_factorial.py')
Invoke-Stage -Name 'W09_ARRAYS' -Arguments @('src\analysis\materialize_w09_validation_arrays.py')
Invoke-Stage -Name 'W10_SWEEP' -Arguments @('src\generators\run_w10_prior_sweep.py', '--run-all')
Invoke-Stage -Name 'W10_ANALYSIS' -Arguments @('src\analysis\analyze_w10_mechanisms.py')
Invoke-Stage -Name 'W11_DIFFUSION' -Arguments @('src\generators\run_w11_diffusion.py', '--run-all')

$receipt = [ordered]@{
    schema_version = 1
    status = 'COMPLETE'
    completed_at = (Get-Date).ToString('o')
    stages = @('W09_ANALYSIS', 'W09_ARRAYS', 'W10_SWEEP', 'W10_ANALYSIS', 'W11_DIFFUSION')
    test_payload_accessed = $false
}
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logRoot 'completion.json') -Encoding utf8
