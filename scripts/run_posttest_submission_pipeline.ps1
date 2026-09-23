param(
    [int]$WaitForProcessId = 0,
    [double]$TestSessionHours = 10.5,
    [int]$CooldownMinutes = 30
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$scientificPython = Join-Path $projectRoot '.venv-w05\Scripts\python.exe'
$reportingPython = Join-Path $projectRoot '.venv-reporting\Scripts\python.exe'
$logRoot = Join-Path $projectRoot 'runs\new_study\posttest_submission_pipeline'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Invoke-Stage {
    param([string]$Name, [string]$Executable, [string[]]$Arguments)
    Write-Output "START $Name $((Get-Date).ToString('o'))"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Output "COMPLETE $Name $((Get-Date).ToString('o'))"
}

function Get-ReceiptStatus {
    param([string]$RelativePath)
    $path = Join-Path $projectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path)) { return 'MISSING' }
    return (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).status
}

function Invoke-Cooldown {
    if ($CooldownMinutes -le 0) { return }
    Write-Output "COOLDOWN_START minutes=$CooldownMinutes $((Get-Date).ToString('o'))"
    Start-Sleep -Seconds ($CooldownMinutes * 60)
    Write-Output "COOLDOWN_COMPLETE $((Get-Date).ToString('o'))"
}

Set-Location -LiteralPath $projectRoot
if ($WaitForProcessId -gt 0) {
    $process = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-Output "WAIT PRETEST_PIPELINE PID=$WaitForProcessId"
        Wait-Process -Id $WaitForProcessId
    }
}

Invoke-Cooldown
Invoke-Stage -Name 'W17_PREFLIGHT' -Executable $scientificPython -Arguments @('src\evaluation\run_locked_test_campaign.py', '--preflight')
while ((Get-ReceiptStatus -RelativePath 'evidence\test_campaign\W17_TEST_COMPLETENESS.json') -ne 'COMPLETE') {
    Invoke-Stage -Name 'W17_TEST_SESSION' -Executable $scientificPython -Arguments @(
        'src\evaluation\run_locked_test_campaign.py', '--run-all', '--session-hours',
        $TestSessionHours.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    $status = Get-ReceiptStatus -RelativePath 'evidence\test_campaign\W17_TEST_COMPLETENESS.json'
    Write-Output "W17_TEST receipt_status=$status $((Get-Date).ToString('o'))"
    if ($status -eq 'PAUSED_COMPUTE_CEILING_GUARD') {
        throw 'W17_TEST stopped at the cumulative compute ceiling guard'
    }
    if ($status -ne 'COMPLETE') { Invoke-Cooldown }
}

Invoke-Stage -Name 'W17_STATISTICS_PREFLIGHT' -Executable $scientificPython -Arguments @('src\analysis\run_w17_statistics.py', '--preflight')
Invoke-Stage -Name 'W17_STATISTICS' -Executable $scientificPython -Arguments @('src\analysis\run_w17_statistics.py', '--run')
Invoke-Stage -Name 'W18_SUBMISSION_PREFLIGHT' -Executable $reportingPython -Arguments @('src\reporting\build_submission_artifacts.py', '--preflight')
Invoke-Stage -Name 'W18_SUBMISSION_BUILD' -Executable $reportingPython -Arguments @('src\reporting\build_submission_artifacts.py', '--build')
Invoke-Stage -Name 'W21_PDF_BUILD' -Executable 'powershell' -Arguments @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'scripts\build_manuscript.ps1', '-RenderForQA')
Invoke-Stage -Name 'W20_INTERNAL_AUDIT_PREVISUAL' -Executable $reportingPython -Arguments @('src\reporting\audit_submission.py', '--run')

$receipt = [ordered]@{
    schema_version = 1
    status = 'READY_FOR_FINAL_VISUAL_QA_AND_RELEASE'
    completed_at = (Get-Date).ToString('o')
    stages = @('W17_TEST', 'W17_STATISTICS', 'W18_SUBMISSION_BUILD', 'W21_PDF_BUILD', 'W20_INTERNAL_AUDIT_PREVISUAL')
    expected_pdf_count = 3
    independent_evaluator = 'EXCLUDED_BY_USER_SCOPE'
}
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logRoot 'completion.json') -Encoding utf8
Write-Output "READY_FOR_FINAL_VISUAL_QA_AND_RELEASE $((Get-Date).ToString('o'))"
