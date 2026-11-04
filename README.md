# RAG vs No-RAG in Transplant Infectious Disease

Code for a 39-model evaluation of a hybrid retrieval-augmented generation (RAG) configuration versus direct (No-RAG) generation on 21 transplant infectious disease cases, with LLM-judge scoring, masked clinician ratings, and a retrieval audit.

Manuscript: "Hybrid Retrieval-Augmented Generation Was Associated With Lower Clinical Reasoning Quality in Transplant Infectious Disease" (PLOS Digital Health, PDIG-D-26-01326, in revision). The comparison is between two complete configurations, which differed in prompts and answer format as well as retrieval.

Data: https://doi.org/10.5281/zenodo.23892698. The datasets are archived on Zenodo, not in this repository.

## Contents

### Generation and judging

| File | Description |
|---|---|
| `run_bedrock_model_with_RAG_metrics.py` | Hybrid RAG pipeline (BM25 + SapBERT + BGE reranker) for AWS Bedrock models; computes generation-time grounding metrics. |
| `run_bedrock_model_without_RAG.py` | No-RAG generation for Bedrock models. |
| `run_ollama_model_with_RAG_metrics.py` | Hybrid RAG pipeline for open-weight models served through Ollama. |
| `run_ollama_model_NO_RAG.py` | No-RAG generation for Ollama models. |
| `judge_all_models_multidim_csv.py` | Six-dimension LLM-judge scoring of model outputs (OpenAI, Gemini, and Anthropic judges). |
| `error_taxonomy_irr.py` | Agreement between the keyword-coded judge-rationale categories and a second LLM coder (Cohen's κ). |
| `document_processor.py` | PDF ingestion, chunking (1,500 characters, 100 overlap), and embedding of the 226-document corpus. |
| `bedrock_manager.py`, `config.py` | Bedrock API wrapper and system configuration. |
| `complex_prompts.json`, `simple_prompts.json` | The acute-management (A-01 to A-10) and surveillance/specialized (S-01 to S-11) case prompts. |
| `setup_ollama_aws.md`, `setup_ollama_ec2.sh`, `submit_job_index_papers.sh` | Ollama setup on EC2 and corpus indexing on HPC. |

These scripts were not rerun or modified for the revision.

### Analysis

| File | Description |
|---|---|
| `finalize_statistics.py` | Main analysis: judge ratings averaged within each answer, two-way model/case clustered inference, Benjamini–Hochberg adjustment of the six dimensions, retrieval subsets, and case-level exact sign-flip tests for the expert ratings. |
| `revision_addenda.py` | Same-developer judge sensitivity analysis and the S-07 leflunomide unit counts. |
| `prepare_release_data.py` | Deidentified expert ratings (E1–E5), case-level differences, and copies of the case prompts. |
| `build_supplement_data.py`, `export_supplement_csvs.py` | S1–S4 table matrices and their CSV export. |
| `make_manuscript_figs.py` | The seven 300-dpi TIFF figures (Arial, labels at least 8 pt). Figure 7 reads `s3_source.csv`. |
| `build_prompt_supplement.py` | Extracts the generation and judge prompt templates from this repository into S6. |
| `inspect_retrieved_passages.py` | Verifies retrieved passages against the exact-text hashes in the Zenodo manifest, given authorized access to the document cache. |
| `run_core.ps1` | Runs the analysis scripts in order. |
| `RAG_TxID_analysis_FULL.R` | Original-submission pipeline. Its significance tests treated the three judge ratings of each answer as independent and are superseded by `finalize_statistics.py`; its eleven-case filter also selected the wrong subset. It is kept because it produced the judge-rationale category counts used for S3 and Figure 7. |

## Reproduce

The analysis scripts read inputs from a project folder containing `data/` (the Zenodo version 1 files), `revision_statistics/`, and the submitted supplementary tables. Set `RAG_PROJECT_DIR` to that folder or run the scripts from it. `finalize_statistics.py` also reads the two added evaluators' rating exports, which are not public; the expert analyses can be recomputed from `expert_ratings_deidentified.csv` in the Zenodo release.

```bash
pip install -r requirements.txt
export RAG_PROJECT_DIR=/path/to/project
python finalize_statistics.py
python prepare_release_data.py
python build_supplement_data.py
python export_supplement_csvs.py
python make_manuscript_figs.py
python build_prompt_supplement.py
python revision_addenda.py
```

On Windows: `./run_core.ps1 -Python python -ProjectDir C:\path\to\project`.

To inspect retrieved passages, see `retrieval/PASSAGE_ACCESS_GUIDE.md` in the Zenodo release. `inspect_retrieved_passages.py` takes `--manifest`, either `--cache` or `--passages-json`, `--output-dir`, and optionally `--case S-07`. The document cache is not public.

The original R pipeline (R 4.3 or later with dplyr, tidyr, readr, stringr, forcats, purrr, tibble, ggplot2, scales, ggrepel, patchwork, lme4, lmerTest, and emmeans) runs with `Rscript RAG_TxID_analysis_FULL.R`; set `RAG_DATA_DIR` and `RAG_OUT_DIR` to change the input and output folders.

### Generation and judging

```bash
pip install haystack-ai sentence-transformers boto3 python-dotenv
python run_bedrock_model_with_RAG_metrics.py --model claude-3-sonnet
python run_bedrock_model_without_RAG.py --model claude-3-sonnet
python judge_all_models_multidim_csv.py --input results.csv --outdir judged/ --judge_provider claude --judge_model <judge-model-id>
```

Set AWS credentials through `~/.aws/credentials` or environment variables. See `setup_ollama_aws.md` for the open-weight models.

## Key results

Judge ratings averaged within each answer; standard errors clustered by model and case.

| Finding | Result |
|---|---|
| Composite quality, RAG − No-RAG | −0.355 (95% CI −0.500 to −0.210), *P* < 0.001 |
| Correctness | −0.196 (95% CI −0.423 to 0.030), *P* = 0.086 |
| Mean judge major-error flag | +3.4 percentage points (95% CI −4.4 to 11.2), *P* = 0.375 |
| Models with lower composite quality under RAG | 37/39 |
| Excluding same-developer judge ratings (294 records) | −0.365 (95% CI −0.513 to −0.216) |
| Good-retrieval cases (7/21), exploratory | −0.204 (95% CI −0.540 to 0.132), *P* = 0.188 |
| Excluding severe retrieval failures (11 cases), exploratory | −0.242 (95% CI −0.461 to −0.023), *P* = 0.034 |
| Five evaluators, 10 case pairs (exact sign-flip), accuracy | −0.48, *P* = 0.043 |
| Five evaluators, 10 case pairs (exact sign-flip), usefulness | −0.94, *P* = 0.0078 |
| S-07 answers with a mg/mL leflunomide target | 25/39 RAG, 0/39 No-RAG |

The expert pairs were selected after automated scoring, so the expert results indicate agreement with the LLM judges rather than an unbiased estimate of the configuration effect.
