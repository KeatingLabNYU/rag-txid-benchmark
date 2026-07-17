#!/usr/bin/env python3
"""
Run Bedrock models directly (NO RAG).

This script loads prompts from `simple_prompts.json` and `complex_prompts.json`
and sends each prompt as a direct user query to an AWS Bedrock model using
`BedrockManager.query_model`. Outputs are written to
`results_bedrock_NO_RAG/` as both text and JSON.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from bedrock_manager import BedrockManager, BedrockModels, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


CLINICIAN_SYSTEM_PROMPT = """You are a transplant infectious diseases medical assistant.

Answer for a clinician audience using clear, scannable prose.

Requirements:
- Do not use tables.
- Do not expose chain-of-thought or internal reasoning.
- Use short section headers when helpful.
- Prefer concise paragraphs and bullet points over long dense blocks.
- State uncertainty briefly when evidence is limited.
- Keep the answer practical and clinically readable.

Default structure:
1. Clinical assessment
2. Recommended management
3. Monitoring and follow-up
4. Key cautions or tradeoffs

If the prompt asks for diagnostics, differential diagnosis, prophylaxis, drug interactions, or immunosuppression changes, include those as clearly labeled sections.
"""


def load_all_prompts(simple_path: str = "simple_prompts.json", complex_path: str = "complex_prompts.json") -> list:
    base_dir = Path(__file__).resolve().parent
    simple_file = base_dir / simple_path
    complex_file = base_dir / complex_path

    if not simple_file.exists() or not complex_file.exists():
        logger.error("Prompt files not found. Make sure simple_prompts.json and complex_prompts.json exist.")
        sys.exit(1)

    with open(simple_file, "r") as f:
        simple = json.load(f)
    with open(complex_file, "r") as f:
        complex_ = json.load(f)

    prompts = []
    for p in simple:
        prompts.append({"id": p.get("id"), "description": p.get("title", p.get("id", "")), "prompt": p["prompt"]})
    for p in complex_:
        prompts.append({"id": p.get("id"), "description": p.get("title", p.get("id", "")), "prompt": p["prompt"]})

    prompts.sort(key=lambda x: x["id"] or "")
    logger.info(f"Loaded {len(prompts)} prompts (simple + complex)")
    return prompts


def run_direct_bedrock(manager: BedrockManager, model_config: ModelConfig, prompts: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_config.model_name.replace("/", "_").replace(":", "_")
    out_txt = output_dir / f"{safe_name}_results.txt"
    out_json = output_dir / f"{safe_name}_results.json"

    results = []

    with open(out_txt, "w") as tf:
        tf.write(f"Model: {model_config.model_name}\n")
        tf.write(f"Model ID: {model_config.model_id}\n")
        tf.write(f"Timestamp: {datetime.now().isoformat()}\n")
        tf.write("=" * 80 + "\n\n")

    for i, p in enumerate(prompts, 1):
        print(f"Query {i}/{len(prompts)} → {p.get('id')}: {p.get('description')}")
        resp = manager.query_model(
            model_config,
            query=p["prompt"],
            context="",
            system_prompt=CLINICIAN_SYSTEM_PROMPT,
        )

        entry = {
            "id": p.get("id"),
            "description": p.get("description"),
            "prompt": p.get("prompt"),
            "success": resp.get("success", False),
            "answer": resp.get("answer", ""),
            "error": resp.get("error") if not resp.get("success", False) else None,
            "latency_seconds": resp.get("latency_seconds", None),
        }
        results.append(entry)

        with open(out_txt, "a") as tf:
            tf.write(f"ID: {entry['id']}\n")
            tf.write(f"Prompt: {entry['prompt']}\n")
            tf.write(f"Latency: {entry['latency_seconds']}\n")
            tf.write(f"Success: {entry['success']}\n")
            if entry['success']:
                tf.write(f"Answer:\n{entry['answer']}\n\n")
            else:
                tf.write(f"Error: {entry['error']}\n\n")
            tf.write("-" * 60 + "\n\n")

        # save progress after each query
        with open(out_json, "w") as jf:
            json.dump({"model": model_config.model_name, "model_id": model_config.model_id, "timestamp": datetime.now().isoformat(), "results": results}, jf, indent=2)

    print(f"Saved text output → {out_txt}")
    print(f"Saved JSON output → {out_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bedrock models directly (no RAG). If no --model-id provided, the script will run the verified Bedrock models list.")
    parser.add_argument("--model-id", type=str, help="Bedrock model id to run (e.g., meta.llama3-8b-instruct-v1:0). If omitted, runs default model list.")
    parser.add_argument("--all", action="store_true", help="Explicitly run all verified Bedrock models from bedrock_manager.BedrockModels")

    parser.add_argument("--model-name", type=str, default=None, help="Optional friendly model name for outputs")
    parser.add_argument("--profile", type=str, default="shreyas", help="AWS profile name (defaults to shreyas)")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--output-dir", type=str, default="results_bedrock_NO_RAG", help="Directory to save outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_all_prompts()

    manager = BedrockManager(region=args.region, profile_name=args.profile)

    # Determine which models to run:
    # 1) If --model-id provided, run that single model
    # 2) Else if --all provided or no flags provided, run the verified BedrockModels list
    models_to_run = []
    if args.model_id:
        mc = ModelConfig(model_id=args.model_id, model_name=args.model_name or args.model_id, provider="Bedrock")
        models_to_run = [mc]
    else:
        # default to the same models used by run_bedrock_model_with_RAG_metrics.py
        models_to_run = BedrockModels.get_all_models() if (args.all or not args.model_id) else []

    out_dir = Path(args.output_dir)
    for model in models_to_run:
        print(f"\nRunning model: {model.model_name} [{model.model_id}]\n")
        run_direct_bedrock(manager, model, prompts, out_dir)


if __name__ == "__main__":
    main()
