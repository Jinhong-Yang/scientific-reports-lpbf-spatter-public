param(
    [string]$OutputDirectory = "output/pdf",
    [switch]$RenderForQA
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$manuscript = Join-Path $root "manuscript"
$build = Join-Path $manuscript "build"
$output = Join-Path $root $OutputDirectory
New-Item -ItemType Directory -Force -Path $build, $output | Out-Null

Push-Location $manuscript
try {
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Initial main manuscript LaTeX pass failed" }
    & bibtex build/main | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Main manuscript BibTeX pass failed" }
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Second main manuscript LaTeX pass failed" }
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Final main manuscript LaTeX pass failed" }
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build supplementary.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Supplementary Information LaTeX pass failed" }
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build supplementary.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Final Supplementary Information LaTeX pass failed" }
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build cover_letter.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Cover letter LaTeX pass failed" }
}
finally {
    Pop-Location
}

$outputs = @{
    (Join-Path $build "main.pdf") = (Join-Path $output "Scientific_Reports_LPBF_Manuscript.pdf")
    (Join-Path $build "supplementary.pdf") = (Join-Path $output "Scientific_Reports_LPBF_Supplementary_Information.pdf")
    (Join-Path $build "cover_letter.pdf") = (Join-Path $output "Scientific_Reports_LPBF_Cover_Letter.pdf")
}
foreach ($entry in $outputs.GetEnumerator()) {
    Copy-Item -LiteralPath $entry.Key -Destination $entry.Value -Force
}

$warningPattern = "undefined|LaTeX Warning|Overfull|Underfull"
$warnings = foreach ($name in "main", "supplementary", "cover_letter") {
    $log = Join-Path $build "$name.log"
    Select-String -LiteralPath $log -Pattern $warningPattern | ForEach-Object {
        [pscustomobject]@{document=$name; line=$_.LineNumber; message=$_.Line.Trim()}
    }
}

if ($RenderForQA) {
    $qaRoot = Join-Path $root "tmp/pdfs/final"
    New-Item -ItemType Directory -Force -Path $qaRoot | Out-Null
    foreach ($entry in $outputs.GetEnumerator()) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($entry.Value)
        $destination = Join-Path $qaRoot $stem
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        & pdftoppm -png -r 150 $entry.Value (Join-Path $destination "page") | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "PDF rendering failed for $($entry.Value)" }
    }
}

[pscustomobject]@{
    status = if ($warnings) { "BUILT_WITH_WARNINGS" } else { "PASS" }
    outputs = @($outputs.Values)
    warnings = @($warnings)
} | ConvertTo-Json -Depth 5
