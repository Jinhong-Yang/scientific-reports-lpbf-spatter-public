param(
    [int]$ParentProcessId = 0,
    [double]$DetectorSessionHours = 3.5
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-w05\Scripts\python.exe'
$logRoot = Join-Path $projectRoot 'runs\new_study\post_w11_pipeline'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Invoke-Stage {
    param([string]$Name, [string[]]$Arguments)
    Write-Output "START $Name $((Get-Date).ToString('o'))"
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Output "COMPLETE $Name $((Get-Date).ToString('o'))"
}

Set-Location -LiteralPath $projectRoot
if ($ParentProcessId -gt 0) {
    $process = Get-Process -Id $ParentProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-Output "WAIT PRETEST_GENERATOR_PIPELINE PID=$ParentProcessId"
        Wait-Process -Id $ParentProcessId
    }
}

$diffusion = Get-Content -LiteralPath (Join-Path $projectRoot 'evidence\stronger_generator\W11_DIFFUSION_COMPLETENESS.json') -Raw | ConvertFrom-Json
if ($diffusion.status -ne 'COMPLETE' -or $diffusion.completed -ne 10 -or $diffusion.failed_or_interrupted -ne 0) {
    throw 'W11 diffusion did not reach immutable 10/10 completion'
}

Invoke-Stage -Name 'W11_POOLS' -Arguments @('src\generators\build_w11_pools.py', '--run-all')
Invoke-Stage -Name 'W11_QUALITY' -Arguments @('src\analysis\analyze_w11_quality.py', '--run')
Invoke-Stage -Name 'W12_PREFLIGHT' -Arguments @('src\detectors\run_w12_detectors.py', '--preflight')
Invoke-Stage -Name 'W12_CORE_SESSION' -Arguments @('src\detectors\run_w12_detectors.py', '--run-all', '--session-hours', $DetectorSessionHours.ToString([Globalization.CultureInfo]::InvariantCulture))

$receipt = [ordered]@{
    schema_version = 1
    status = 'SESSION_COMPLETE'
    completed_at = (Get-Date).ToString('o')
    stages = @('W11_POOLS', 'W11_QUALITY', 'W12_PREFLIGHT', 'W12_CORE_SESSION')
    detector_session_hours = $DetectorSessionHours
    test_payload_accessed = $false
}
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logRoot 'completion.json') -Encoding utf8
