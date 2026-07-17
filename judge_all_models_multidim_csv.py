#!/usr/bin/env python3
"""
judge_all_models_multidim_csv.py

Same logic as your multi-dim judge script, but **reads your CSV output format**.

Input CSV must contain at least these columns:
- Model
- ID
- Prompt
- Answer

Optional columns (if present, carried through to outputs):
- Description
- Latency (s)
- GK Used
- GMKR
- Retrieval Alignment
- Evidence Density
- Distinct Docs Cited
- Total Citations
- Sources   (highly recommended; otherwise judge will often flag "uses_outside_knowledge")

Outputs (in --outdir):
- __judged.json
- __judged.csv
- __summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Data model
# -----------------------------
DIMENSION_KEYS = [
    "correctness",
    "completeness",
    "reasoning",
    "relevance",
    "clarity",
    "consistency",
]

JUDGE_REQUIRED_KEYS = set(DIMENSION_KEYS) | {
    "is_refusal",
    "has_major_error",
    "major_error_summary",
    "has_minor_error",
    "minor_error_summary",
    "uses_outside_knowledge",
    "outside_knowledge_summary",
}


@dataclass
class ParsedItem:
    model_name: str
    timestamp: str
    item_id: str
    query_title: str
    prompt: str
    answer: str

    # optional metrics from upstream
    latency_s: Optional[float] = None
    uses_general_knowledge: Optional[str] = None
    gmkr: Optional[float] = None
    retrieval_alignment_score: Optional[float] = None
    evidence_density_per_100t: Optional[float] = None
    rur_distinct_docs: Optional[int] = None
    rur_total_citations: Optional[int] = None

    sources_raw: str = ""


# -----------------------------
# Parsing: CSV
# -----------------------------
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _to_boolish_yesno(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"yes", "y", "true", "1"}:
            return "Yes"
        if s in {"no", "n", "false", "0"}:
            return "No"
        return None
    if isinstance(x, (int, float)):
        return "Yes" if bool(x) else "No"
    if isinstance(x, bool):
        return "Yes" if x else "No"
    return None


def parse_results_csv(path: Path) -> List[ParsedItem]:
    # Use pandas if available for robustness; fall back to csv module.
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    items: List[ParsedItem] = []

    if pd is not None:
        df = pd.read_csv(path)
        required = {"Model", "ID", "Prompt", "Answer"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for _, r in df.iterrows():
            model = str(r["Model"]).strip()
            item_id = str(r["ID"]).strip()
            prompt = str(r["Prompt"]).strip()
            answer = str(r["Answer"]).strip()

            query_title = str(r.get("Description", "")).strip()
            sources_raw = str(r.get("Sources", "")).strip()

            items.append(
                ParsedItem(
                    model_name=model,
                    timestamp=str(r.get("Timestamp", "")).strip(),
                    item_id=item_id,
                    query_title=query_title,
                    prompt=prompt,
                    answer=answer,
                    latency_s=_to_float(r.get("Latency (s)", None)),
                    uses_general_knowledge=_to_boolish_yesno(r.get("GK Used", None)),
                    gmkr=_to_float(r.get("GMKR", None)),
                    retrieval_alignment_score=_to_float(r.get("Retrieval Alignment", None)),
                    evidence_density_per_100t=_to_float(r.get("Evidence Density", None)),
                    rur_distinct_docs=_to_int(r.get("Distinct Docs Cited", None)),
                    rur_total_citations=_to_int(r.get("Total Citations", None)),
                    sources_raw=sources_raw,
                )
            )
        return items

    # Fallback without pandas
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        required = {"Model", "ID", "Prompt", "Answer"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            items.append(
                ParsedItem(
                    model_name=str(row.get("Model", "")).strip(),
                    timestamp=str(row.get("Timestamp", "")).strip(),
                    item_id=str(row.get("ID", "")).strip(),
                    query_title=str(row.get("Description", "")).strip(),
                    prompt=str(row.get("Prompt", "")).strip(),
                    answer=str(row.get("Answer", "")).strip(),
                    latency_s=_to_float(row.get("Latency (s)", None)),
                    uses_general_knowledge=_to_boolish_yesno(row.get("GK Used", None)),
                    gmkr=_to_float(row.get("GMKR", None)),
                    retrieval_alignment_score=_to_float(row.get("Retrieval Alignment", None)),
                    evidence_density_per_100t=_to_float(row.get("Evidence Density", None)),
                    rur_distinct_docs=_to_int(row.get("Distinct Docs Cited", None)),
                    rur_total_citations=_to_int(row.get("Total Citations", None)),
                    sources_raw=str(row.get("Sources", "")).strip(),
                )
            )

    return items


# -----------------------------
# Prompt building / helpers
# -----------------------------
def sanitize_model_name(name: str) -> str:
    name = name.lower().strip()
    name = name.replace(" ", "")
    name = name.replace(".", "_")
    return name


def extract_answer_section(text: str) -> str:
    # If answer contains "Answer:" header (from other formats), strip it.
    m = re.search(r"(?is)\banswer\s*:\s*(.*)$", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def build_judge_system() -> str:
    return (
        "You are a strict clinical QA evaluator. "
        "Return ONLY valid JSON. No extra keys. No markdown."
    )


def build_judge_user(prompt: str, answer: str, sources_raw: str) -> str:
        sources_block = sources_raw.strip()
        if not sources_block:
                sources_block = "(none listed)"

        # NOTE: This is intentionally similar to a typical multi-dim rubric but focus only on answer content.
        return f"""You will evaluate a model answer to a clinical question. Evaluate ONLY the text provided below and do not penalize or reward citation fidelity, source formatting, or whether references match verbatim (even if source snippets are provided).

CLINICAL QUESTION / PROMPT:
{prompt}

MODEL ANSWER:
{answer}

RETRIEVED SOURCE LIST (if provided; ignore fidelity):
{sources_block}

Score each dimension from 1 to 5:
- correctness: factual/clinical accuracy. 5 = fully correct, 1 = largely incorrect regardless of references.
- completeness: addresses all key parts; includes appropriate differential/steps if needed.
- reasoning: logical clinical reasoning and justification; avoids leaps.
- relevance: stays on-topic; avoids irrelevant tangents.
- clarity: well-structured, readable, unambiguous.
- consistency: no internal contradictions; consistent recommendations.

Also provide booleans:
- is_refusal: true if the model refuses to answer.
- has_major_error: true if there is a dangerous/major clinical error; summarize briefly.
- has_minor_error: true if there is an observable but minor clinical misstep; summarize.

Return ONLY JSON with EXACTLY these keys:
{{
    "correctness": int,
    "completeness": int,
    "reasoning": int,
    "relevance": int,
    "clarity": int,
    "consistency": int,
    "is_refusal": bool,
    "has_major_error": bool,
    "major_error_summary": str,
    "has_minor_error": bool,
    "minor_error_summary": str,
}}
"""


def validate_judgment(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Judge output is not a JSON object.")

    # Guard against missing Gemini keys that frequently arrive empty
    obj.setdefault("uses_outside_knowledge", False)
    obj.setdefault("outside_knowledge_summary", "")
    obj.setdefault("has_major_error", False)
    obj.setdefault("has_minor_error", False)
    obj.setdefault("major_error_summary", "")
    obj.setdefault("minor_error_summary", "")

    keys = set(obj.keys())
    missing = JUDGE_REQUIRED_KEYS - keys
    extra = keys - JUDGE_REQUIRED_KEYS
    if missing:
        raise ValueError(f"Judge JSON missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Judge JSON has extra keys: {sorted(extra)}")

    for k in DIMENSION_KEYS:
        v = obj[k]
        if not isinstance(v, int):
            raise ValueError(f"{k} must be int 1-5, got {type(v)}")
        if v < 1 or v > 5:
            raise ValueError(f"{k} out of range 1-5: {v}")

    for b in ["is_refusal", "has_major_error", "has_minor_error", "uses_outside_knowledge"]:
        if not isinstance(obj[b], bool):
            raise ValueError(f"{b} must be boolean")

    for s in ["major_error_summary", "minor_error_summary", "outside_knowledge_summary"]:
        val = obj[s]
        if val is None:
            obj[s] = ""
        elif not isinstance(val, str):
            obj[s] = str(val)

    return obj


def robust_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract first {...} block
    m = re.search(r"(?s)\{.*\}", text)
    if not m:
        raise ValueError("Could not find JSON object in judge output.")
    return json.loads(m.group(0))


# -----------------------------
# Judge providers
# -----------------------------
def judge_openai(
    judge_model: str,
    system: str,
    user: str,
    api_key: Optional[str],
    base_url: Optional[str],
) -> Dict[str, Any]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    kwargs: Dict[str, Any] = {
        "model": judge_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }

    # gpt-5 family uses max_completion_tokens in some SDKs
    if judge_model.lower().startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 900
    else:
        kwargs["max_tokens"] = 900

    kwargs["temperature"] = 0

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    return robust_json_loads(text)


def judge_gemini(
    judge_model: str,
    system: str,
    user: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    try:
        import google.genai as genai  # type: ignore  # new google-genai package
        backend = "genai"
    except Exception:
        import google.generativeai as genai  # type: ignore  # fallback for older google-generativeai
        backend = "generativeai"

    if not api_key:
        raise ValueError("Gemini requires --api_key or GEMINI_API_KEY env var.")

    if backend == "genai":
        # --- New SDK (google-genai v1.0+) ---
        from google.genai import types # type: ignore
        
        client = genai.Client(api_key=api_key)

        def _genai_call(model_name: str):
            cfg = {
                "system_instruction": system,
                "temperature": 0,
                "response_mime_type": "application/json",
            }
            if types is not None:
                cfg = types.GenerateContentConfig(**cfg)

            return client.models.generate_content(
                model=model_name,
                contents=user,
                config=cfg,
            )

        # Try without prefix first, then fallback
        try:
            resp = _genai_call(judge_model)
        except Exception:
            model_id = judge_model if judge_model.startswith("models/") else f"models/{judge_model}"
            resp = _genai_call(model_id)

        text = getattr(resp, "text", "") or ""
    
    else:
        # --- Old SDK (google-generativeai v0.x) ---
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=judge_model,
            system_instruction=system,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )
        resp = model.generate_content(user)
        text = getattr(resp, "text", "") or ""

    return robust_json_loads(text)


def judge_claude(
    judge_model: str,
    system: str,
    user: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    from anthropic import Anthropic  # type: ignore

    if not api_key:
        raise ValueError("Claude requires --api_key or ANTHROPIC_API_KEY env var.")
    client = Anthropic(api_key=api_key)

    resp = client.messages.create(
        model=judge_model,
        system=system,
        max_tokens=900,
        temperature=0,
        messages=[{"role": "user", "content": user}],
    )
    # Anthropic returns structured blocks; join any text blocks
    chunks = []
    for b in resp.content:
        if getattr(b, "type", None) == "text":
            chunks.append(b.text)
    text = "\n".join(chunks).strip()
    return robust_json_loads(text)


def run_judge(
    provider: str,
    model: str,
    system: str,
    user: str,
    api_key: Optional[str],
    base_url: Optional[str],
) -> Dict[str, Any]:
    p = provider.lower()
    if p == "openai":
        return judge_openai(model, system, user, api_key, base_url)
    if p == "gemini":
        return judge_gemini(model, system, user, api_key)
    if p == "claude":
        return judge_claude(model, system, user, api_key)
    raise ValueError(f"Unknown judge_provider: {provider}")


# -----------------------------
# Summaries
# -----------------------------
def mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]  # type: ignore
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_by_model(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r.get("model_name", "UNKNOWN"), []).append(r)

    summary: Dict[str, Any] = {}
    for model, rs in by_model.items():
        # mean of each dimension ignoring None
        means = {}
        for k in DIMENSION_KEYS:
            scores = [r.get(k) for r in rs if isinstance(r.get(k), (int, float))]
            means[k] = (sum(scores) / len(scores)) if scores else None

        n = len(rs)
        refusal_rate = (sum(1 for r in rs if r.get("is_refusal") is True) / n) if n else None
        major_error_rate = (sum(1 for r in rs if r.get("has_major_error") is True) / n) if n else None
        outside_knowledge_rate = (sum(1 for r in rs if r.get("uses_outside_knowledge") is True) / n) if n else None
        high_correct_rate = (
            sum(1 for r in rs if isinstance(r.get("correctness"), int) and r["correctness"] >= 4) / n
            if n
            else None
        )

        summary[model] = {
            "n": n,
            "mean_scores": means,
            "refusal_rate": refusal_rate,
            "major_error_rate": major_error_rate,
            "outside_knowledge_rate": outside_knowledge_rate,
            "high_correctness_rate_(>=4)": high_correct_rate,
        }

    return summary


# -----------------------------
# Summary CSV helpers
# -----------------------------
def summary_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in sorted(summary.keys()):
        info = summary[model]
        row: Dict[str, Any] = {
            "model": model,
            "n": info.get("n"),
        }
        means = info.get("mean_scores", {}) or {}
        for dim in DIMENSION_KEYS:
            row[f"{dim}_mean"] = means.get(dim)
        row["refusal_rate"] = info.get("refusal_rate")
        row["major_error_rate"] = info.get("major_error_rate")
        row["outside_knowledge_rate"] = info.get("outside_knowledge_rate")
        row["high_correctness_rate_(>=4)"] = info.get("high_correctness_rate_(>=4)")
        rows.append(row)
    return rows


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    rows = summary_to_rows(summary)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = [
        "model",
        "n",
        *[f"{dim}_mean" for dim in DIMENSION_KEYS],
        "refusal_rate",
        "major_error_rate",
        "outside_knowledge_rate",
        "high_correctness_rate_(>=4)",
    ]

    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# -----------------------------
# IO helpers
# -----------------------------
def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    # union of all keys (stable ordering)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV file with results")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--judge_provider", required=True, choices=["openai", "gemini", "claude"])
    ap.add_argument("--judge_model", required=True, help="Judge model name")
    ap.add_argument("--api_key", default=None, help="API key (or use env vars)")
    ap.add_argument("--base_url", default=None, help="Optional base_url for OpenAI-compatible endpoints")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Sleep between judge calls (rate limiting)")
    ap.add_argument("--max_items", type=int, default=None, help="Limit items (debug)")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Provider-specific default env keys
    api_key = args.api_key
    if not api_key:
        if args.judge_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif args.judge_provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif args.judge_provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")

    print(f"📖 Parsing CSV: {in_path}")
    items = parse_results_csv(in_path)
    if args.max_items:
        items = items[: args.max_items]

    if not items:
        print(f"❌ No items parsed from {in_path}")
        sys.exit(2)

    system_msg = build_judge_system()
    judged_rows: List[Dict[str, Any]] = []

    # Prepare output paths
    provider = args.judge_provider.lower()
    provider_map = {
        "openai": "openai",
        "gemini": "google",
        "claude": "anthropic",
    }
    provider = provider_map.get(provider, provider)

    model_tag = sanitize_model_name(args.judge_model)
    stem = in_path.stem

    judged_csv_path = outdir / f"{stem}_judged_{provider}_{model_tag}.csv"
    judged_json_path = outdir / f"{stem}_judged_{provider}_{model_tag}.json"
    summary_path = outdir / f"{stem}_summary_{provider}_{model_tag}.json"
    per_model_summary_path = outdir / f"{stem}_per_model_summary.csv"

    for i, it in enumerate(items, start=1):
        print(f"🧪 [{i}/{len(items)}] Judging: model={it.model_name} id={it.item_id}")

        answer_to_judge = extract_answer_section(it.answer)
        user_msg = build_judge_user(it.prompt, answer_to_judge, it.sources_raw)

        row = asdict(it)
        row.setdefault("has_minor_error", False)
        row.setdefault("minor_error_summary", "")
        try:
            raw = run_judge(
                provider=args.judge_provider,
                model=args.judge_model,
                system=system_msg,
                user=user_msg,
                api_key=api_key,
                base_url=args.base_url,
            )
            judged = validate_judgment(raw)
            row["_judge_raw"] = json.dumps(raw, ensure_ascii=False)
            row.update(judged)
        except Exception as e:
            # keep row, mark judge fields as None and store error message
            for k in DIMENSION_KEYS:
                row[k] = None
            row["is_refusal"] = None
            row["has_major_error"] = True
            row["major_error_summary"] = f"JUDGE_ERROR: {type(e).__name__}: {e}"
            row["has_minor_error"] = False
            row["minor_error_summary"] = ""
            row["uses_outside_knowledge"] = None
            row["outside_knowledge_summary"] = ""
        judged_rows.append(row)

        # Write incremental results after each judgment
        write_json(judged_json_path, judged_rows)
        write_csv(judged_csv_path, judged_rows)
        
        # Update summary after each result
        summary = summarize_by_model(judged_rows)
        write_json(summary_path, summary)
        write_summary_csv(per_model_summary_path, summary)

        if args.sleep_s and args.sleep_s > 0:
            time.sleep(args.sleep_s)

    print("\n✅ Done.")
    print(f"  - {judged_json_path}")
    print(f"  - {judged_csv_path}")
    print(f"  - {summary_path}")

    # Print quick summary
    print("\n📊 Per-model summary:")
    for model, s in summary.items():
        means = s["mean_scores"]
        means_str = ", ".join(f"{k}={means[k]:.2f}" for k in DIMENSION_KEYS if means.get(k) is not None)
        print(f"  - {model}: n={s['n']} | {means_str}")


if __name__ == "__main__":
    main()