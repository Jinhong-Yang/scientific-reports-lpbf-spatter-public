param([switch]$SkipTests, [switch]$RebuildManuscript)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv-reporting\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = 'python'
}
Set-Location -LiteralPath $projectRoot

if (-not $SkipTests) {
    & $python -m pytest -q tests\test_submission_artifacts.py tests\test_submission_audit.py tests\test_release_archive.py
    if ($LASTEXITCODE -ne 0) { throw 'Public-package test suite failed' }
}

if (Test-Path -LiteralPath 'release\SHA256SUMS.txt') {
    & $python scripts\verify_sha256s.py release\SHA256SUMS.txt
    if ($LASTEXITCODE -ne 0) { throw 'Release checksum verification failed' }
}

if ($RebuildManuscript) {
    & $python src\reporting\build_submission_artifacts.py --build
    if ($LASTEXITCODE -ne 0) { throw 'Manuscript tables and figures regeneration failed' }
}

Write-Output 'Reproducibility checks complete.'
