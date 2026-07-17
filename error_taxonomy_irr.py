"""
Error Taxonomy Inter-Rater Reliability (IRR) Script
====================================================
Sends each major_error_summary through a second LLM judge (Claude via Anthropic API)
using the same 8-category rubric, then computes Cohen's kappa against the original coding.

Usage:
    pip install anthropic scikit-learn --break-system-packages
    export ANTHROPIC_API_KEY="sk-ant-..."
    python error_taxonomy_irr.py \
        --input  ../all_models_rag_non_rag_judged.csv \
        --output error_taxonomy_irr_results.csv

The output CSV has columns:
    row_index, model_name, item_id, rag, original_category, rater2_category, match

Cohen's kappa is printed to stdout; also saved in error_taxonomy_irr_kappa.txt.
"""

import argparse
import csv
import json
import time
import sys
from pathlib import Path

# ── Category definitions (same rubric as original coding) ─────────────────
CATEGORIES = [
    "Source misattribution",
    "Ignores explicit case facts",
    "Critical omission",
    "Dosing / quantitative",
    "Drug interaction",
    "Outdated guidance",
    "Incorrect/unsafe management",
    "Other / unclassified",
]

SYSTEM_PROMPT = """\
You are a clinical error classifier. You will be given a brief description of a clinical error
made by a large language model answering a transplant infectious disease question.

Classify the error into EXACTLY ONE of the following categories (respond with the category
label only — no explanation, no punctuation, just the label):

1. Source misattribution — error caused by misreading or over-relying on a retrieved source
   (e.g., reproducing a typo from a document, citing a source for a claim it does not support)
2. Ignores explicit case facts — model ignores or contradicts information explicitly stated
   in the clinical vignette (e.g., wrong organ transplant type, ignores stated allergy)
3. Critical omission — important clinical element absent from the answer
   (e.g., missed diagnosis, missing drug, missing monitoring parameter, incomplete differential)
4. Dosing / quantitative — incorrect dose, frequency, duration, units, or numeric threshold
5. Drug interaction — fails to identify or incorrectly characterizes a clinically important
   drug–drug or drug–pathogen interaction
6. Outdated guidance — recommends a management approach superseded by current guidelines
7. Incorrect/unsafe management — incorrect treatment recommendation not covered by the above
   categories (e.g., wrong drug class, contraindicated therapy)
8. Other / unclassified — error does not fit any of the above categories

Output ONLY the category label from this list. Do not include a number prefix."""

USER_TEMPLATE = "Error description: {summary}"

def classify_one(client, summary: str, max_retries: int = 3) -> str:
    """Call Claude and return the classified category label."""
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",   # fast + cheap for classification
                max_tokens=30,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(summary=summary)}],
            )
            raw = msg.content[0].text.strip()
            # Fuzzy match to known categories
            for cat in CATEGORIES:
                if cat.lower() in raw.lower() or raw.lower() in cat.lower():
                    return cat
            # If no match, return raw (will be counted as Other)
            print(f"  [WARN] Unmapped response: '{raw}' — treating as Other", file=sys.stderr)
            return "Other / unclassified"
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [RETRY {attempt+1}] {e}", file=sys.stderr)
                time.sleep(2 ** attempt)
            else:
                raise

def compute_kappa(labels_a, labels_b, categories):
    """Cohen's kappa (unweighted)."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    n = len(labels_a)
    assert n == len(labels_b), "Label lists must be same length"

    # Observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement (product of marginal frequencies)
    from collections import Counter
    freq_a = Counter(labels_a)
    freq_b = Counter(labels_b)
    p_e = sum((freq_a.get(c, 0) / n) * (freq_b.get(c, 0) / n) for c in categories)

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0
    return kappa, p_o, p_e

def main():
    parser = argparse.ArgumentParser(description="Error taxonomy IRR via second LLM judge")
    parser.add_argument("--input",  required=True, help="Path to publication CSV")
    parser.add_argument("--output", default="error_taxonomy_irr_results.csv")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Limit to N rows (0 = all). Use for testing.")
    args = parser.parse_args()

    # ── Load Anthropic client ──────────────────────────────────────────────
    try:
        import anthropic
    except ImportError:
        sys.exit("Install anthropic: pip install anthropic --break-system-packages")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    # ── Read CSV, filter to error rows ────────────────────────────────────
    print("Reading CSV …", file=sys.stderr)
    error_rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row.get("has_major_error", "").strip().lower() == "true":
                summary = row.get("major_error_summary", "").strip()
                if summary:
                    error_rows.append({
                        "row_index": i,
                        "model_name": row.get("model_name", ""),
                        "item_id": row.get("item_id", ""),
                        "rag": row.get("RAG_MODEL", ""),
                        "summary": summary,
                    })

    if args.limit:
        error_rows = error_rows[:args.limit]

    print(f"  {len(error_rows)} error rows to classify", file=sys.stderr)

    # ── Original coding (rule-based priority classifier, same as R script) ─
    # Priority order: source misattribution → ignores facts → critical omission
    # → dosing → drug interaction → outdated → incorrect → other
    KEYWORDS = {
        "Source misattribution": ["source", "misattribu", "doc ", "document", "retriev", "cited", "passage", "typo"],
        "Ignores explicit case facts": ["ignor", "contradict", "overlook", "miss", "stated", "vignette", "case fact"],
        "Critical omission": ["omit", "omission", "miss", "absent", "fail to mention", "incomplete", "neglect", "lack"],
        "Dosing / quantitative": ["dos", "dose", "mg", "mcg", "unit", "frequency", "duration", "quantit", "numeric", "trough"],
        "Drug interaction": ["drug interaction", "interaction", "ddI", "drug-drug"],
        "Outdated guidance": ["outdated", "supersed", "old guideline", "former", "obsolete"],
        "Incorrect/unsafe management": ["incorrect", "unsafe", "wrong", "contraindic", "inappropriate"],
    }
    PRIORITY = [
        "Source misattribution", "Ignores explicit case facts", "Critical omission",
        "Dosing / quantitative", "Drug interaction", "Outdated guidance",
        "Incorrect/unsafe management",
    ]

    def classify_original(summary: str) -> str:
        s = summary.lower()
        for cat in PRIORITY:
            if any(kw in s for kw in KEYWORDS[cat]):
                return cat
        return "Other / unclassified"

    # ── Classify all rows with rater 2 (LLM) ──────────────────────────────
    results = []
    for idx, row in enumerate(error_rows, 1):
        orig = classify_original(row["summary"])
        print(f"  [{idx}/{len(error_rows)}] classifying …", end="\r", file=sys.stderr)
        r2 = classify_one(client, row["summary"])
        results.append({
            "row_index":        row["row_index"],
            "model_name":       row["model_name"],
            "item_id":          row["item_id"],
            "rag":              row["rag"],
            "summary":          row["summary"],
            "original_category": orig,
            "rater2_category":  r2,
            "match":            orig == r2,
        })
        time.sleep(0.1)   # gentle rate limiting

    print(f"\n  Done.", file=sys.stderr)

    # ── Write results CSV ──────────────────────────────────────────────────
    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["row_index","model_name","item_id","rag","summary",
                      "original_category","rater2_category","match"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {out_path}")

    # ── Cohen's kappa ──────────────────────────────────────────────────────
    orig_labels = [r["original_category"] for r in results]
    r2_labels   = [r["rater2_category"]   for r in results]
    kappa, p_o, p_e = compute_kappa(orig_labels, r2_labels, CATEGORIES)

    kappa_txt = (
        f"Error Taxonomy IRR Results\n"
        f"==========================\n"
        f"N (error rows classified): {len(results)}\n"
        f"Observed agreement (P_o):  {p_o:.4f} ({p_o*100:.1f}%)\n"
        f"Expected agreement (P_e):  {p_e:.4f}\n"
        f"Cohen's kappa:             {kappa:.4f}\n\n"
        f"Interpretation: <0.20 slight, 0.20–0.40 fair, 0.41–0.60 moderate,\n"
        f"                0.61–0.80 substantial, >0.80 almost perfect\n"
    )

    # Per-category breakdown
    from collections import Counter
    kappa_txt += "\nPer-category agreement:\n"
    for cat in CATEGORIES:
        both_cat    = sum(a == cat and b == cat for a, b in zip(orig_labels, r2_labels))
        orig_n      = orig_labels.count(cat)
        r2_n        = r2_labels.count(cat)
        kappa_txt  += f"  {cat:35s} orig={orig_n:3d}  rater2={r2_n:3d}  both={both_cat:3d}\n"

    print("\n" + kappa_txt)
    kappa_path = out_path.parent / "error_taxonomy_irr_kappa.txt"
    kappa_path.write_text(kappa_txt)
    print(f"Kappa summary written to {kappa_path}")

if __name__ == "__main__":
    main()
