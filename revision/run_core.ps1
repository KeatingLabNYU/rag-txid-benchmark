param([string]$Python = 'python')

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $projectRoot
$steps = @(
    'finalize_statistics.py',
    'prepare_release_data.py',
    'build_supplement_data.py',
    'export_supplement_csvs.py',
    'build_figures.py',
    'build_prompt_supplement.py',
    'revision_addenda.py'
)
foreach ($step in $steps) {
    Write-Host "Running $step"
    & $Python (Join-Path $PSScriptRoot $step)
    if ($LASTEXITCODE -ne 0) { throw "$step failed with exit code $LASTEXITCODE" }
}
Write-Host 'Core revision results, figures, deidentified data, and portable tables reproduced.'
