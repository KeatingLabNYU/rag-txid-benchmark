param([string]$Python = 'python', [string]$ProjectDir = (Get-Location).Path)

# ProjectDir is the folder holding data/, revision_statistics/ and the other analysis inputs.
Set-Location -LiteralPath $ProjectDir
$env:RAG_PROJECT_DIR = (Resolve-Path $ProjectDir).Path
$steps = @(
    'finalize_statistics.py',
    'prepare_release_data.py',
    'build_supplement_data.py',
    'export_supplement_csvs.py',
    'make_manuscript_figs.py',
    'build_prompt_supplement.py',
    'revision_addenda.py'
)
foreach ($step in $steps) {
    Write-Host "Running $step"
    & $Python (Join-Path $PSScriptRoot $step)
    if ($LASTEXITCODE -ne 0) { throw "$step failed with exit code $LASTEXITCODE" }
}
Write-Host 'Core revision results, figures, deidentified data, and portable tables reproduced.'
