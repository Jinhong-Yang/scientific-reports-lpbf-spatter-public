param(
    [double]$CombinedStageDetectorSessionHours = 8.0,
    [int]$CooldownMinutes = 30
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-w05\Scripts\python.exe'
$logRoot = Join-Path $projectRoot 'runs\new_study\remaining_pretest_pipeline'
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

function Get-ReceiptStatus {
    param([string]$RelativePath)
    $path = Join-Path $projectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path)) {
        return 'MISSING'
    }
    return (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).status
}

function Assert-ReceiptStatus {
    param(
        [string]$RelativePath,
        [string]$Name,
        [string[]]$AcceptedStatuses = @('COMPLETE')
    )
    $status = Get-ReceiptStatus -RelativePath $RelativePath
    if ($status -notin $AcceptedStatuses) {
        throw "$Name must have one of the accepted statuses before resume: $($AcceptedStatuses -join ', ')"
    }
}

function Invoke-Cooldown {
    Write-Output "COOLDOWN_START minutes=$CooldownMinutes $((Get-Date).ToString('o'))"
    Start-Sleep -Seconds ($CooldownMinutes * 60)
    Write-Output "COOLDOWN_COMPLETE $((Get-Date).ToString('o'))"
}

function Invoke-DetectorSessions {
    param(
        [string]$Name,
        [string]$Script,
        [string]$Receipt
    )
    while ((Get-ReceiptStatus -RelativePath $Receipt) -ne 'COMPLETE') {
        Invoke-Stage -Name "$Name`_SESSION" -Arguments @($Script, '--run-all', '--session-hours', $CombinedStageDetectorSessionHours.ToString([Globalization.CultureInfo]::InvariantCulture))
        $status = Get-ReceiptStatus -RelativePath $Receipt
        Write-Output "$Name receipt_status=$status $((Get-Date).ToString('o'))"
        if ($status -eq 'COMPLETE') {
            break
        }
        if ($status -eq 'PAUSED_COMPUTE_CEILING_GUARD') {
            throw "$Name stopped at the cumulative compute ceiling guard"
        }
        Invoke-Cooldown
    }
}

Set-Location -LiteralPath $projectRoot
Assert-ReceiptStatus -RelativePath 'evidence\detector_core\CORE_COMPLETENESS.json' -Name 'W12_CORE'
Assert-ReceiptStatus -RelativePath 'evidence\resolution\W15_COMPLETENESS.json' -Name 'W15_RESOLUTION' -AcceptedStatuses @('PASS')
Assert-ReceiptStatus -RelativePath 'evidence\low_data\W14_GENERATOR_COMPLETENESS.json' -Name 'W14_GENERATORS'

Invoke-DetectorSessions -Name 'W14_DETECTORS' -Script 'src\detectors\run_w14_lowdata_detectors.py' -Receipt 'evidence\low_data\W14_DETECTOR_COMPLETENESS.json'
Invoke-Stage -Name 'W14_FULL_INTEGRITY_AUDIT' -Arguments @('scripts\verify_w14_integrity.py')

Invoke-Cooldown
Invoke-Stage -Name 'W16_GENERATOR_PREFLIGHT' -Arguments @('src\generators\run_w16_grouped_generators.py', '--preflight')
Invoke-Stage -Name 'W16_GENERATORS' -Arguments @('src\generators\run_w16_grouped_generators.py', '--run-all')
Assert-ReceiptStatus -RelativePath 'evidence\grouped_internal\W16_GENERATOR_COMPLETENESS.json' -Name 'W16_GENERATORS'
Invoke-Stage -Name 'W16_DETECTOR_PREFLIGHT' -Arguments @('src\detectors\run_w16_grouped_detectors.py', '--preflight')
Invoke-DetectorSessions -Name 'W16_DETECTORS' -Script 'src\detectors\run_w16_grouped_detectors.py' -Receipt 'evidence\grouped_internal\W16_DETECTOR_COMPLETENESS.json'
Invoke-Stage -Name 'W16_FULL_INTEGRITY_AUDIT' -Arguments @('scripts\verify_w16_integrity.py')

$receipt = [ordered]@{
    schema_version = 1
    status = 'PRETEST_EXECUTION_COMPLETE'
    completed_at = (Get-Date).ToString('o')
    resumed_from = 'W14_DETECTORS'
    stages = @('W14_DETECTORS', 'W14_FULL_INTEGRITY_AUDIT', 'W16_GENERATORS', 'W16_DETECTORS', 'W16_FULL_INTEGRITY_AUDIT')
    maximum_combined_stage_detector_session_hours = $CombinedStageDetectorSessionHours
    cooldown_minutes = $CooldownMinutes
    test_payload_accessed = $false
}
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logRoot 'completion.json') -Encoding utf8
Write-Output "PRETEST_EXECUTION_COMPLETE $((Get-Date).ToString('o'))"
