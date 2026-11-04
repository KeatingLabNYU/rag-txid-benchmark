# Reproduce the revision analysis

> **Note for this repository.** These scripts are identical to `revision_scripts/` in the Zenodo release (https://doi.org/10.5281/zenodo.23892698). Paths below refer to the authors' project folder. The `retrieval/` files referenced here, including `PASSAGE_ACCESS_GUIDE.md`, are in the Zenodo release. The evaluator rating exports and other internal inputs are not public; the released ratings are labeled E1–E5.

## Inspect retrieved evidence locally

Start with `../retrieval/PASSAGE_ACCESS_GUIDE.md`. `inspect_retrieved_passages.py` reads either an authorized original document cache (`--cache`; requires NumPy) or an authorized recovered export (`--passages-json`; standard library only), verifies exact-text hashes from `--manifest`, and writes a local HTML reader and JSON to `--output-dir`. Add `--case S-07` for a single case. The full cache/export is not in the public release. Source publication links, unresolved access, alternative candidates, and manual audit corrections are documented in the guide.

These are the updated scripts used for the new PLOS Digital Health revision files. Run them from the project folder that contains `data/`, the two new expert CSV exports, the submitted S1/S3/S4 workbooks, and `rag-txid-benchmark-main/`. The original generation and judge scripts were **not rerun or modified**. The revision's statistical analysis is implemented in Python because the supplied legacy R script does not contain the final output-level clustered analyses and its eleven-case filter selects the wrong subset.

## One-command run on Windows

With Python 3.10 or newer and the packages in `requirements.txt` installed:

```powershell
Set-Location 'D:\Plos_Digital_Health_Revision'
python -m pip install -r revision_submission\scripts\requirements.txt
& revision_submission\scripts\run_core.ps1 -Python python
```

The runner executes these scripts in order:

1. `finalize_statistics.py` creates all output-averaged, two-way model/case clustered results, six-dimension BH adjustments, ten-case exact sign-flip human results, subset analyses, and input hashes in `revision_statistics/`. It uses NumPy and pandas, not SciPy or R. **This internal analysis folder contains evaluator names** for source matching and should not be uploaded as the public expert dataset.
2. `prepare_release_data.py` creates 100 deidentified expert ratings, 50 evaluator-case differences, ten case means, exact case-prompt copies, and the source counts used for Figure 7 in `revision_submission/data/`.
3. `build_supplement_data.py` creates the typed S1–S4 matrices; `export_supplement_csvs.py` writes eight portable tables under `revision_submission/data/tables_csv/`.
4. `build_figures.py` creates the seven separate 300-dpi TIFF figures submitted with the revision (Arial, labels at least 8 pt) in `revision_submission/figures/`. Figure 7 reads the category rates in `s3_source.csv`, which sits next to the script.
5. `build_prompt_supplement.py` extracts the current Bedrock, Ollama, and judge prompt templates into S6.
6. `revision_addenda.py` prints the judge same-developer sensitivity analysis (Section 3.6) and the S-07 leflunomide unit counts (Section 3.7).

The formatted S1–S4 `.xlsx` files were built from the same matrices with `build_supplements.mjs`. The CSVs are the portable numeric source if that library is unavailable. The `build_revised_manuscript.py` script creates the clean Word draft from the submitted DOCX; the marked Word copy was made with Microsoft Word's Compare Documents feature. Those two manuscript-generation steps are separate from the statistical runner.

## Checks and interpretation

- Core quality: −0.354920525458 (95% CI −0.50028929838 to −0.209551752537).
- Core correctness: −0.196411032597 (95% CI −0.422985836411 to 0.0301637712173).
- Mean judge major-error flag difference: +0.0339825437625 (95% CI −0.0440680583009 to 0.112033145826).
- Five-evaluator exact case-level P values: 0.04296875 for accuracy and 0.0078125 for usefulness.
- Excluding same-developer judge ratings (294 records): quality −0.365 (95% CI −0.513 to −0.216).
- S-07 answers giving a mg/mL leflunomide target: 25 of 39 RAG, 0 of 39 No-RAG.
- Seven good-retrieval cases and eleven cases excluding severe document-level failures are defined explicitly in `revision_statistics/analysis_manifest.json`.

The scripts do not generate new model answers, new expert evaluations, clinical adjudication, or proof that retrieval alone caused the observed difference.
