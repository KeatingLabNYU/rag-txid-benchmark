#!/usr/bin/env python3
"""
Run RAG benchmarks with AWS Bedrock models (simple + complex prompts) while
tracking RAG-specific quality metrics:
 - GMKR (General Medical Knowledge Ratio)
 - Retrieval Alignment Score (Synonym-Aware Jaccard overlap)
 - RUR (Retrieval Utilization Rate)
 - Evidence Density

UPDATED: 
1. Includes Medical Synonym Expansion.
2. HOTFIX V3: Handles API quirks for Amazon Nova AND Cohere Command R.
   - Nova: renames 'max_tokens' -> 'maxTokens', removes 'top_p'
   - Cohere: removes 'top_p' (keeps 'max_tokens')
"""

import argparse
import logging
import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime
import time
import pickle

# Haystack Imports
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.builders import ChatPromptBuilder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.document_stores.types import DuplicatePolicy
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

# Project imports
from config import SystemConfig
from bedrock_manager import ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# METRIC UTILITIES (MEDICALLY AWARE)
# =============================================================================

STOPWORDS = {
    "the", "and", "is", "of", "in", "to", "for", "with", "a", "an", 
    "on", "at", "by", "from", "that", "which", "are", "it", "be", 
    "as", "or", "not", "this", "we", "can", "has", "have"
}

MEDICAL_SYNONYMS = {
    "renal": "kidney", "kidney": "renal",
    "hepatic": "liver", "liver": "hepatic",
    "cardiac": "heart", "heart": "cardiac",
    "pulmonary": "lung", "lung": "pulmonary",
    "nephrotoxicity": "kidney", 
    "mycotic": "fungal", "fungal": "mycotic",
    "bkv": "bk", "bk": "bkv",
    "cytomegalovirus": "cmv", "cmv": "cytomegalovirus",
    "ebv": "epsteinbarr", "epsteinbarr": "ebv",
    "pjp": "pneumocystis", "pneumocystis": "pjp",
    "clostridioides": "cdiff", "clostridium": "cdiff", "difficile": "cdiff",
    "is": "immunosuppression", "immunosuppression": "is",
    "tac": "tacrolimus", "tacrolimus": "tac",
    "mmf": "mycophenolate", "mycophenolate": "mmf",
    "t-cell": "tcell", "tcell": "t-cell"
}

MODEL_MAX_TOKENS_LIMITS = {
    # Models with documented hard caps
    "ai21.jamba-1-5-mini-v1:0": 4096,
    "meta.llama3-70b-instruct-v1:0": 2048,
    "meta.llama3-8b-instruct-v1:0": 2048,
}

def clamp_max_tokens(model_id: str, requested: int) -> int:
    cap = MODEL_MAX_TOKENS_LIMITS.get(model_id.lower())
    if cap is None or requested <= cap:
        return requested
    logger.warning(
        "Requested %d tokens for %s exceeds the hard cap of %d; using %d tokens",
        requested,
        model_id,
        cap,
        cap,
    )
    return cap

def tokenize(text: str):
    clean_text = re.sub(r'[^\w\s-]', '', text.lower())
    raw_tokens = set(clean_text.split())
    tokens = raw_tokens - STOPWORDS
    expanded_tokens = set(tokens)
    for t in tokens:
        if t in MEDICAL_SYNONYMS:
            expanded_tokens.add(MEDICAL_SYNONYMS[t])
    return expanded_tokens

def calculate_retrieval_alignment(answer, context_chunks, top_n: int = 3):
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    chunks = context_chunks[:top_n] if len(context_chunks) >= top_n else context_chunks
    overlaps = []
    for ch in chunks:
        if hasattr(ch, "content"):
            chunk_text = ch.content
        elif isinstance(ch, dict):
            chunk_text = ch.get("content", "")
        else:
            chunk_text = str(ch)
        chunk_tokens = tokenize(chunk_text)
        intersection = len(answer_tokens & chunk_tokens)
        union = len(answer_tokens | chunk_tokens) or 1
        overlaps.append(intersection / union)
    return max(overlaps) if overlaps else 0.0

# =============================================================================
# BEDROCK RAG SYSTEM
# =============================================================================

class BedrockRAGSystem:
    """RAG system using AWS Bedrock for generation."""

    def __init__(self):
        self.document_store = InMemoryDocumentStore()

    def load_documents_from_cache(
        self, cache_path: str = "haystack_store/cached_documents.pkl"
    ):
        cache_file = Path(cache_path)
        if not cache_file.exists():
            logger.error(f"Cache file not found: {cache_path}")
            sys.exit(1)

        logger.info(f"📂 Loading cached documents from {cache_path}...")
        try:
            with open(cache_file, "rb") as f:
                cached_docs = pickle.load(f)
            self.document_store.write_documents(
                cached_docs, policy=DuplicatePolicy.SKIP
            )
            logger.info(f"✅ Loaded {len(cached_docs)} documents from cache")
            return len(cached_docs)
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")
            sys.exit(1)

    def create_hybrid_pipeline(
        self, model_config: ModelConfig, temperature: float = None
    ) -> Pipeline:
        pipeline = Pipeline()

        pipeline.add_component(
            "query_embedder",
            SentenceTransformersTextEmbedder(model=SystemConfig.EMBEDDING_MODEL),
        )
        pipeline.add_component(
            "bm25_retriever",
            InMemoryBM25Retriever(document_store=self.document_store),
        )
        pipeline.add_component(
            "embedding_retriever",
            InMemoryEmbeddingRetriever(document_store=self.document_store),
        )
        pipeline.add_component("joiner", DocumentJoiner(join_mode="concatenate"))
        pipeline.add_component(
            "ranker",
            TransformersSimilarityRanker(
                model="BAAI/bge-reranker-v2-m3",
                top_k=5,
            ),
        )

        template = """
    You are a medical AI assistant specializing in transplant infectious diseases.
    Your task is to provide accurate, comprehensive, and safe medical answers.

    PROTOCOL (follow strictly):

    1. DOCUMENTS FIRST:
       Answer using ONLY the information found in the provided "Context from research papers".
       Cite using: [Doc X].

    2. GAP FILLING:
       If a needed detail is missing (e.g., comparator drug or dosing guidance),
       you may include general medical knowledge, BUT you must:
          - Start a new paragraph labeled exactly: "[General Knowledge / Non-Guideline]"
          - Explicitly state that this paragraph is not from the retrieved documents.
          - Do NOT add citations to that paragraph.

    3. CITATIONS:
       - Cite only the documents shown below.
       - NEVER fabricate citations.
       - General-knowledge paragraphs must not include citations.

    4. NO DISCLAIMERS:
       Do NOT include empathy, apologies, or professional disclaimers.

    5. STRUCTURE:
       Provide your answer *exactly* in this format:

       **Direct Answer**
       <2–6 sentence concise answer prioritizing document evidence>

       **Explanation**
       <Medical reasoning referencing the documents; insert the labeled general-knowledge paragraph only if needed>

       **Citations**
       <List of relevant Doc numbers, e.g., [Doc 1, Doc 3]>

    -------------------------------------------------------
    Context from research papers:
    {% for doc in documents %}
    [Doc {{ loop.index }}] (Title: {{ doc.meta.get('title', doc.meta['filename']) }}):
    {{ doc.content }}
    {% endfor %}

    Question: {{ query }}

    Answer:
    """

        chat_template = [ChatMessage.from_user(template)]
        pipeline.add_component(
            "chat_prompt_builder",
            ChatPromptBuilder(
                template=chat_template,
                required_variables=["documents", "query"],
            ),
        )

        # ---------------------------------------------------------------------
        # HOTFIX V3 FOR BEDROCK API COMPATIBILITY (Nova & Cohere)
        # ---------------------------------------------------------------------
        
        # Base settings
        temp_val = temperature if temperature is not None else model_config.temperature
        effective_max_tokens = clamp_max_tokens(
            model_config.model_id, model_config.max_tokens
        )
        model_id_lower = model_config.model_id.lower()
        
        # 1. Parameter Naming (Nova enforces camelCase)
        if "nova" in model_id_lower:
            gen_kwargs = {
                "maxTokens": effective_max_tokens,  # RENAME for Nova
                "temperature": temp_val
            }
        else:
            gen_kwargs = {
                "max_tokens": effective_max_tokens, # STANDARD for others
                "temperature": temp_val
            }

        # 2. Top-P Exclusion (Nova AND Cohere reject it via Converse)
        if "nova" not in model_id_lower and "cohere" not in model_id_lower:
             gen_kwargs["top_p"] = model_config.top_p

        generator = AmazonBedrockChatGenerator(
            model=model_config.model_id,
            generation_kwargs=gen_kwargs,
        )
        pipeline.add_component("generator", generator)

        pipeline.connect(
            "query_embedder.embedding", "embedding_retriever.query_embedding"
        )
        pipeline.connect("bm25_retriever", "joiner")
        pipeline.connect("embedding_retriever", "joiner")
        pipeline.connect("joiner.documents", "ranker.documents")
        pipeline.connect("ranker", "chat_prompt_builder.documents")
        pipeline.connect("chat_prompt_builder.prompt", "generator.messages")

        return pipeline

    def query(self, pipeline: Pipeline, query: str, top_k: int = 5):
        start = time.time()
        candidate_k = top_k * 2

        run_params = {
            "query_embedder": {"text": query},
            "bm25_retriever": {"query": query, "top_k": candidate_k},
            "embedding_retriever": {"top_k": candidate_k},
            "ranker": {"query": query, "top_k": top_k},
            "chat_prompt_builder": {"query": query},
        }

        result = pipeline.run(
            run_params, include_outputs_from={"ranker", "chat_prompt_builder"}
        )

        latency = time.time() - start

        reply = result["generator"]["replies"][0]
        if hasattr(reply, "text"):
            answer = reply.text
        elif hasattr(reply, "content"):
            content = reply.content
            if isinstance(content, str):
                answer = content
            elif isinstance(content, list) and content:
                first = content[0]
                if hasattr(first, "text"):
                    answer = first.text
                elif isinstance(first, dict) and "text" in first:
                    answer = first["text"]
                else:
                    answer = str(first)
            else:
                answer = str(content)
        else:
            answer = str(reply)

        docs = []
        if "ranker" in result and "documents" in result["ranker"]:
            docs = result["ranker"]["documents"]

        sources = []
        for i, doc in enumerate(docs, 1):
            title = doc.meta.get("title", doc.meta.get("filename", "Unknown"))
            sources.append(
                {
                    "index": i,
                    "title": title,
                    "filename": doc.meta.get("filename", "Unknown"),
                    "score": getattr(doc, "score", doc.meta.get("score", 0.0)),
                    "preview": (
                        doc.content[:150] + "..."
                        if isinstance(doc.content, str) and len(doc.content) > 150
                        else doc.content
                    ),
                }
            )

        uses_general_knowledge = "[General Knowledge / Non-Guideline]" in answer
        gmkr = 0.0
        gk_tokens = 0
        rag_tokens = 0

        if uses_general_knowledge:
            marker = "[General Knowledge / Non-Guideline]"
            parts = answer.split(marker, 1)
            if len(parts) == 2:
                rag_part, gk_part = parts
                rag_tokens = len(rag_part.split())
                gk_tokens = len(gk_part.split())
                total_tokens = rag_tokens + gk_tokens
                if total_tokens > 0:
                    gmkr = gk_tokens / total_tokens

        retrieval_alignment = calculate_retrieval_alignment(answer, docs, top_n=3)

        citations_match = re.search(
            r"\*\*Citations\*\*\s*\n\s*(\[Doc[^\]]+\](?:\s*,\s*\[Doc[^\]]+\])*)",
            answer,
            re.IGNORECASE,
        )
        if citations_match:
            citations_text = citations_match.group(1)
            citations = re.findall(r"\[Doc\s*\d+\]", citations_text)
        else:
            citations = re.findall(r"\[Doc\s*\d+\]", answer)

        n_distinct_docs = len(set(citations))
        n_citations_raw = len(citations)
        rur_flag = 1 if n_distinct_docs > 0 else 0

        all_citations = re.findall(r"\[Doc\s*\d+\]", answer)
        answer_tokens = answer.split()
        n_tokens = len(answer_tokens) if answer_tokens else 1
        evidence_density_100 = (len(all_citations) / n_tokens) * 100

        return {
            "answer": answer,
            "context_chunks": docs,
            "sources": sources,
            "latency_seconds": latency,
            "success": True,
            "uses_general_knowledge": uses_general_knowledge,
            "gmkr": round(gmkr, 4),
            "rag_tokens": rag_tokens,
            "gk_tokens": gk_tokens,
            "retrieval_alignment": round(retrieval_alignment, 4),
            "rur_flag": rur_flag,
            "n_distinct_docs": n_distinct_docs,
            "n_citations_raw": n_citations_raw,
            "n_tokens": n_tokens,
            "evidence_density": round(evidence_density_100, 2),
        }


def load_all_prompts(
    simple_path: str = "simple_prompts.json", complex_path: str = "complex_prompts.json"
) -> list:
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

    prompts.sort(key=lambda x: x["id"] or "")
    logger.info(f"✅ Loaded {len(prompts)} prompts (simple + complex)")
    return prompts


def benchmark_model(rag_system: BedrockRAGSystem, model_config: ModelConfig, prompts: list):
    print(f"\n🤖 Model: {model_config.model_name} [{model_config.model_id}]")
    print(f"📊 Running {len(prompts)} prompts (simple + complex)\n")

    try:
        pipeline = rag_system.create_hybrid_pipeline(model_config)
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return

    results = []

    output_dir = Path("results_bedrock_with_RAG_METRICS")
    output_dir.mkdir(exist_ok=True)
    safe_model_name = model_config.model_name.replace(":", "_").replace("/", "_")
    output_txt = output_dir / f"{safe_model_name}_results.txt"
    output_json = output_dir / f"{safe_model_name}_results.json"

    with open(output_txt, "w") as f:
        f.write(f"Model: {model_config.model_name}\n")
        f.write(f"Model ID: {model_config.model_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

    for i, prompt in enumerate(prompts, 1):
        desc = f"{prompt.get('id', f'Q{i}')}: {prompt['description']}"
        print(f"   🔎 Query {i}/{len(prompts)} → {desc}")
        try:
            result = rag_system.query(pipeline, prompt["prompt"])

            print("\n-------------------")
            print(f"ID: {prompt.get('id', f'Q{i}')}")
            print(f"📌 Prompt: {prompt['prompt']}")
            print(f"🧠 Answer:\n{result['answer']}")
            print(f"⏱ Latency: {result['latency_seconds']:.2f} sec")
            print(f"📚 Docs Used: {len(result['context_chunks'])}")

            if result["sources"]:
                print("\n📄 Sources:")
                for src in result["sources"]:
                    print(
                        f"   [{src['index']}] {src['title']} "
                        f"(score: {src['score']:.3f})"
                    )
            print("-------------------\n")

            answer_text = result["answer"]
            citations_match = re.search(
                r"\*\*Citations\*\*\s*(.+?)(?:\n\n|\Z)",
                answer_text,
                re.IGNORECASE | re.DOTALL,
            )
            if citations_match:
                citations_text = citations_match.group(1)
                doc_numbers = re.findall(r"Doc\s*(\d+)", citations_text)
                n_distinct_docs = len(set(doc_numbers))
                n_citations_raw = len(doc_numbers)
            else:
                doc_numbers = re.findall(r"Doc\s*(\d+)", answer_text)
                n_distinct_docs = len(set(doc_numbers))
                n_citations_raw = len(doc_numbers)

            rur_flag = 1 if n_distinct_docs > 0 else 0

            answer_tokens = answer_text.split()
            n_tokens = len(answer_tokens) if answer_tokens else 1
            evidence_density = (n_citations_raw / n_tokens) * 100

            result_entry = {
                "id": prompt.get("id"),
                "description": prompt["description"],
                "prompt": prompt["prompt"],
                "expected_format": prompt.get("expected_format", []),
                "answer": answer_text,
                "latency": result["latency_seconds"],
                "sources": result["sources"],
                "uses_general_knowledge": result.get("uses_general_knowledge", False),
                "gmkr": result.get("gmkr", 0.0),
                "rag_tokens": result.get("rag_tokens", 0),
                "gk_tokens": result.get("gk_tokens", 0),
                "retrieval_alignment": result.get("retrieval_alignment", 0.0),
                "rur_flag": rur_flag,
                "n_distinct_docs": n_distinct_docs,
                "n_citations_raw": n_citations_raw,
                "n_tokens": n_tokens,
                "evidence_density": round(evidence_density, 2),
            }
            results.append(result_entry)

            with open(output_txt, "a") as f:
                f.write(f"ID: {result_entry['id']}\n")
                f.write(f"Query {i}: {result_entry['description']}\n")
                f.write(f"Prompt: {result_entry['prompt']}\n")
                f.write(f"Latency: {result_entry['latency']:.2f}s\n")
                f.write(
                    "Uses General Knowledge: "
                    f"{'Yes' if result_entry.get('uses_general_knowledge', False) else 'No'}\n"
                )
                if result_entry.get("gmkr", 0.0) > 0:
                    f.write(
                        f"GMKR: {result_entry['gmkr']:.4f} "
                        f"(RAG: {result_entry.get('rag_tokens', 0)} tokens, "
                        f"GK: {result_entry.get('gk_tokens', 0)} tokens)\n"
                    )
                f.write(
                    f"Retrieval Alignment Score: "
                    f"{result_entry.get('retrieval_alignment', 0.0):.4f}\n"
                )
                f.write(
                    f"RUR: {result_entry.get('n_distinct_docs', 0)} distinct docs cited "
                    f"({result_entry.get('n_citations_raw', 0)} total citations)\n"
                )
                f.write(
                    "Evidence Density: "
                    f"{result_entry.get('evidence_density', 0.0):.2f} citations per "
                    f"100 tokens ({result_entry.get('n_tokens', 0)} total tokens)\n"
                )
                f.write("\n")
                f.write(f"Answer:\n{result_entry['answer']}\n\n")
                f.write("Sources:\n")
                for src in result_entry["sources"]:
                    f.write(f"  [{src['index']}] {src['title']}\n")
                    f.write(f"      File: {src['filename']}\n")
                    f.write(f"      Score: {src['score']:.3f}\n")
                f.write("\n" + "=" * 80 + "\n\n")

            gk_count = sum(1 for r in results if r.get("uses_general_knowledge", False))
            gk_percentage = (gk_count / len(results) * 100) if results else 0
            total_gmkr = sum(r.get("gmkr", 0.0) for r in results)
            avg_gmkr = (total_gmkr / len(results)) if results else 0
            gk_queries = [r for r in results if r.get("uses_general_knowledge", False)]
            avg_gmkr_when_used = (
                sum(r.get("gmkr", 0.0) for r in gk_queries) / len(gk_queries)
            ) if gk_queries else 0
            total_ras = sum(r.get("retrieval_alignment", 0.0) for r in results)
            avg_ras = (total_ras / len(results)) if results else 0
            total_rur_flags = sum(r.get("rur_flag", 0) for r in results)
            rur = (total_rur_flags / len(results)) if results else 0
            avg_distinct_docs = (
                sum(r.get("n_distinct_docs", 0) for r in results) / len(results)
                if results
                else 0
            )
            avg_citations = (
                sum(r.get("n_citations_raw", 0) for r in results) / len(results)
                if results
                else 0
            )
            avg_evidence_density = (
                sum(r.get("evidence_density", 0.0) for r in results) / len(results)
                if results
                else 0
            )

            with open(output_json, "w") as jf:
                json.dump(
                    {
                        "model": model_config.model_name,
                        "model_id": model_config.model_id,
                        "timestamp": datetime.now().isoformat(),
                        "summary": {
                            "total_queries": len(results),
                            "general_knowledge_used_count": gk_count,
                            "general_knowledge_percentage": round(gk_percentage, 2),
                            "avg_gmkr_all_queries": round(avg_gmkr, 4),
                            "avg_gmkr_when_used": round(avg_gmkr_when_used, 4),
                            "avg_retrieval_alignment": round(avg_ras, 4),
                            "rur": round(rur, 4),
                            "avg_distinct_docs_cited": round(avg_distinct_docs, 2),
                            "avg_total_citations": round(avg_citations, 2),
                            "avg_evidence_density": round(avg_evidence_density, 2),
                        },
                        "results": results,
                    },
                    jf,
                    indent=2,
                )

        except Exception as e:
            logger.error(f"Query failed for prompt {prompt.get('id', i)}: {e}")
            continue

    if results:
        print(f"\n📄 Saved text output  → {output_txt}")
        print(f"📦 Saved JSON output → {output_json}")

        gk_count = sum(1 for r in results if r.get("uses_general_knowledge", False))
        gk_percentage = (gk_count / len(results) * 100) if results else 0
        total_gmkr = sum(r.get("gmkr", 0.0) for r in results)
        avg_gmkr = (total_gmkr / len(results)) if results else 0
        gk_queries = [r for r in results if r.get("uses_general_knowledge", False)]
        avg_gmkr_when_used = (
            sum(r.get("gmkr", 0.0) for r in gk_queries) / len(gk_queries)
        ) if gk_queries else 0
        total_ras = sum(r.get("retrieval_alignment", 0.0) for r in results)
        avg_ras = (total_ras / len(results)) if results else 0
        total_rur_flags = sum(r.get("rur_flag", 0) for r in results)
        rur = (total_rur_flags / len(results)) if results else 0
        avg_distinct_docs = (
            sum(r.get("n_distinct_docs", 0) for r in results) / len(results)
            if results
            else 0
        )
        avg_citations = (
            sum(r.get("n_citations_raw", 0) for r in results) / len(results)
            if results
            else 0
        )
        avg_evidence_density = (
            sum(r.get("evidence_density", 0.0) for r in results) / len(results)
            if results
            else 0
        )

        print(
            f"📊 General Knowledge Usage: {gk_count}/{len(results)} queries "
            f"({gk_percentage:.1f}%)"
        )
        print(
            f"📊 Average GMKR (all): {avg_gmkr:.4f} | "
            f"GMKR (when used): {avg_gmkr_when_used:.4f}"
        )
        print(f"📊 Average Retrieval Alignment Score: {avg_ras:.4f}")
        print(
            f"📊 RUR (Retrieval Utilization Rate): {rur:.4f} "
            f"({total_rur_flags}/{len(results)} queries cited docs)"
        )
        print(
            f"📊 Avg Citations: {avg_citations:.2f} total | "
            f"{avg_distinct_docs:.2f} distinct docs"
        )
        print(
            f"📊 Evidence Density: {avg_evidence_density:.2f} citations per 100 tokens"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG benchmark with Bedrock models on ALL prompts with RAG metrics"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Bedrock model ID or inference profile ARN (e.g., anthropic.claude-3-sonnet-20240229-v1:0)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Friendly model name for output files (defaults to model-id)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="Bedrock",
        help="Provider label for reporting only",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="haystack_store/cached_documents_sapbert.pkl",
        help="Path to cached documents (use SapBERT-embedded cache)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_config = ModelConfig(
        model_id=args.model_id,
        model_name=args.model_name or args.model_id,
        provider=args.provider,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n🏥 BEDROCK RAG BENCHMARK (ALL simple + complex prompts)\n")
    logger.info(f"🌐 Using Bedrock model: {model_config.model_id}")

    prompts = load_all_prompts()

    print("📚 Loading documents...")
    rag_system = BedrockRAGSystem()
    rag_system.load_documents_from_cache(args.cache)

    benchmark_model(rag_system, model_config, prompts)
    print("\n✨ DONE ✨\n")


if __name__ == "__main__":
    main()