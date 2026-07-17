#!/usr/bin/env python3
"""
Run benchmarks with Ollama models WITHOUT RAG (direct prompts only)
This script queries models directly without document retrieval.
Results are saved with NO_RAG in the filename.
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import time

# Haystack Imports
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


class OllamaDirectSystem:
    """Direct query system using Ollama without RAG"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        """Verify Ollama is accessible"""
        import requests

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Connected to Ollama at {self.ollama_url}")
                models = response.json().get("models", [])
                if models:
                    logger.info(
                        "   Available models: %s",
                        ", ".join([m["name"] for m in models[:5]]),
                    )
            else:
                logger.warning(
                    f"⚠️ Ollama responded with status {response.status_code}"
                )
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Cannot connect to Ollama at {self.ollama_url}")
            logger.error("   Make sure Ollama is running:")
            logger.error("   - Locally: Ollama app/service must be running")
            logger.error("   - On server: `systemctl status ollama` should be active")
            sys.exit(1)
        except Exception as e:
            logger.warning(f"⚠️ Could not verify Ollama connection: {e}")

    def create_direct_pipeline(
        self, model_name: str, temperature: float = 0.1
    ) -> Pipeline:
        """Create direct query pipeline without RAG"""
        pipeline = Pipeline()

        # Simple prompt template for direct medical queries
        template = """
You are a medical AI assistant specializing in transplant infectious diseases.
Your task is to provide accurate, comprehensive, and safe medical answers based on your knowledge.

PROTOCOL (follow strictly):

1. DIRECT ANSWER:
   Provide clear, evidence-based medical information.

2. NO DISCLAIMERS:
   Do NOT include empathy, apologies, or professional disclaimers.

3. STRUCTURE:
   Provide your answer *exactly* in this format:

   **Direct Answer**
   <2–6 sentence concise answer>

   **Explanation**
   <Medical reasoning and detailed information>

Question: {{ query }}

Answer:
"""

        # Chat prompt builder
        chat_template = [ChatMessage.from_user(template)]
        pipeline.add_component(
            "chat_prompt_builder",
            ChatPromptBuilder(
                template=chat_template,
                required_variables=["query"],
            ),
        )

        # Ollama generator
        generator = OllamaChatGenerator(
            model=model_name,
            url=self.ollama_url,
            generation_kwargs={
                "temperature": temperature,
                "num_predict": 4096,
            },
        )
        pipeline.add_component("generator", generator)

        # Connections
        pipeline.connect("chat_prompt_builder.prompt", "generator.messages")

        return pipeline

    def query(self, pipeline: Pipeline, query: str):
        """Run direct query with Ollama model"""
        start = time.time()

        run_params = {
            "chat_prompt_builder": {"query": query},
        }

        result = pipeline.run(run_params)

        latency = time.time() - start

        # Extract answer
        reply = result["generator"]["replies"][0]
        answer = reply.content if hasattr(reply, "content") else str(reply)

        return {
            "answer": answer,
            "latency_seconds": latency,
            "success": True,
        }


def load_all_prompts(
    simple_path: str = "simple_prompts.json", complex_path: str = "complex_prompts.json"
) -> list:
    """
    Load ALL prompts from simple_prompts.json and complex_prompts.json.

    Expected JSON structure:
    [
      {
        "id": "S-01",
        "title": "...",
        "prompt": "...",
        "expected_format": [...]
      },
      ...
    ]
    """
    base_dir = Path(__file__).resolve().parent

    simple_file = base_dir / simple_path
    complex_file = base_dir / complex_path

    if not simple_file.exists():
        logger.error(f"Simple prompts file not found: {simple_file}")
        sys.exit(1)
    if not complex_file.exists():
        logger.error(f"Complex prompts file not found: {complex_file}")
        sys.exit(1)

    with open(simple_file, "r") as f:
        simple = json.load(f)

    with open(complex_file, "r") as f:
        complex_ = json.load(f)

    prompts = []

    for p in simple:
        prompts.append(
            {
                "id": p.get("id"),
                "description": p.get("title", p.get("id", "")),
                "prompt": p["prompt"],
                "expected_format": p.get("expected_format", []),
            }
        )

    for p in complex_:
        prompts.append(
            {
                "id": p.get("id"),
                "description": p.get("title", p.get("id", "")),
                "prompt": p["prompt"],
                "expected_format": p.get("expected_format", []),
            }
        )

    # Sort by ID for deterministic ordering
    prompts.sort(key=lambda x: x["id"] or "")
    logger.info(f"✅ Loaded {len(prompts)} prompts (simple + complex)")
    return prompts


def benchmark_model(direct_system: OllamaDirectSystem, model_name: str, prompts: list):
    """Run benchmark for a single Ollama model without RAG"""

    print(f"\n🤖 Model: {model_name}")
    print(f"📊 Running {len(prompts)} prompts (simple + complex) WITHOUT RAG\n")

    # Create pipeline
    try:
        pipeline = direct_system.create_direct_pipeline(model_name)
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return

    results = []

    for i, prompt in enumerate(prompts, 1):
        desc = f"{prompt.get('id', f'Q{i}')}: {prompt['description']}"
        print(f"   🔎 Query {i}/{len(prompts)} → {desc}")
        try:
            result = direct_system.query(pipeline, prompt["prompt"])

            print("\n-------------------")
            print(f"ID: {prompt.get('id', f'Q{i}')}")
            print(f"📌 Prompt: {prompt['prompt']}")
            print(f"🧠 Answer:\n{result['answer']}")
            print(f"⏱ Latency: {result['latency_seconds']:.2f} sec")
            print("-------------------\n")

            results.append(
                {
                    "id": prompt.get("id"),
                    "description": prompt["description"],
                    "prompt": prompt["prompt"],
                    "expected_format": prompt.get("expected_format", []),
                    "answer": result["answer"],
                    "latency": result["latency_seconds"],
                }
            )

        except Exception as e:
            logger.error(f"Query failed for prompt {prompt.get('id', i)}: {e}")
            continue

    # Save results (text + JSON) with NO_RAG in filename
    if results:
        output_dir = Path("results_ollama")
        output_dir.mkdir(exist_ok=True)

        safe_model_name = model_name.replace(":", "_").replace("/", "_")

        # Human-readable text
        output_txt = output_dir / f"{safe_model_name}_NO_RAG_results.txt"
        with open(output_txt, "w") as f:
            f.write(f"Model: {model_name} (NO RAG)\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            for i, res in enumerate(results, 1):
                f.write(f"ID: {res['id']}\n")
                f.write(f"Query {i}: {res['description']}\n")
                f.write(f"Prompt: {res['prompt']}\n")
                f.write(f"Latency: {res['latency']:.2f}s\n\n")
                f.write(f"Answer:\n{res['answer']}\n\n")
                f.write("\n" + "=" * 80 + "\n\n")

        # Structured JSON (for LLM judge)
        output_json = output_dir / f"{safe_model_name}_NO_RAG_results.json"
        with open(output_json, "w") as jf:
            json.dump(
                {
                    "model": model_name,
                    "mode": "NO_RAG",
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                },
                jf,
                indent=2,
            )

        print(f"📄 Saved text output  → {output_txt}")
        print(f"📦 Saved JSON output → {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark with Ollama models WITHOUT RAG on ALL prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ollama model name (e.g., llama3.1, llama3.1:70b, qwen2.5:14b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama API URL (default: from .env or http://localhost:11434)",
    )

    args = parser.parse_args()

    # Get Ollama URL from args, env, or default
    ollama_url = args.ollama_url or os.getenv(
        "OLLAMA_URL", "http://localhost:11434"
    )
    logger.info(f"🌐 Using Ollama server: {ollama_url}")

    print("\n🏥 OLLAMA DIRECT BENCHMARK (NO RAG - ALL simple + complex prompts)\n")

    # Load prompts
    prompts = load_all_prompts()

    # Initialize direct system
    direct_system = OllamaDirectSystem(ollama_url=ollama_url)

    # Run benchmark
    benchmark_model(direct_system, args.model, prompts)

    print("\n✨ DONE ✨\n")


if __name__ == "__main__":
    main()
