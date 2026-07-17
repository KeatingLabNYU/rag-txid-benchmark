#!/usr/bin/env python3
"""
Run RAG benchmarks with Hugging Face biomedical models.
FIXED VERSION:
- Uses apply_chat_template for correct instruction following.
- Removes destructive repetition penalties that blocked citations.
- Synchronized prompt template with the working Ollama script.
- Improved parsing logic to capture GMKR and Citations correctly.
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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

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

# Project imports
from config import SystemConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def tokenize(text):
    """Simple tokenizer for Retrieval Alignment Score"""
    return set(re.findall(r"\b\w+\b", text.lower()))


def calculate_retrieval_alignment(answer, context_chunks, top_n=3):
    """Calculate Retrieval Alignment Score using Jaccard overlap."""
    answer_tokens = tokenize(answer)
    chunks = context_chunks[:top_n] if len(context_chunks) >= top_n else context_chunks
    
    overlaps = []
    for ch in chunks:
        if hasattr(ch, 'content'):
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


# ---------------------------------------------------------------------------
# Hugging Face biomedical model map
# ---------------------------------------------------------------------------

HF_MODEL_MAP = {
    # Ensure we use Instruct versions where available for better RAG performance
    "biomistral": "BioMistral/BioMistral-7B",
    "openbiollm": "aaditya/Llama3-OpenBioLLM-8B",
    "meditron": "epfl-llm/meditron-7b",
}


def log_gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        logger.info(
            f"🖥 Detected GPU: {props.name} | Total VRAM: ~{total_gb:.1f} GB on cuda:0"
        )
    else:
        logger.warning("⚠️ No CUDA GPU detected. HF model will run on CPU.")


def load_hf_model(model_key: str):
    if model_key not in HF_MODEL_MAP:
        raise ValueError(
            f"Unknown model key '{model_key}'. Choose from: {list(HF_MODEL_MAP.keys())}"
        )

    model_id = HF_MODEL_MAP[model_key]
    logger.info(f"🔬 Loading HF model: {model_id}")

    log_gpu_info()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Fix for models that might miss a chat template configuration
    if not tokenizer.chat_template:
        logger.warning("⚠️ No chat template found in tokenizer. Using standard default.")
        tokenizer.chat_template = "{% for message in messages %}\n{{'<|user|>' if message['role'] == 'user' else '<|assistant|>'}}\n{{ message['content'] }}\n{% endfor %}\n<|assistant|>"

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    logger.info(f"✅ Finished loading HF model: {model_id}")

    return tokenizer, model, model_id


class HFRAGSystem:
    """RAG system using a Hugging Face causal LM for generation"""

    def __init__(
        self,
        model_key: str,
        max_new_tokens: int = 1024, # Increased for detailed answers
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        self.model_key = model_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.document_store = InMemoryDocumentStore()
        self.tokenizer, self.model, self.model_id = load_hf_model(model_key)

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

    def create_hybrid_pipeline(self) -> Pipeline:
        pipeline = Pipeline()

        pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(model=SystemConfig.EMBEDDING_MODEL))
        pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=self.document_store))
        pipeline.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store=self.document_store))
        pipeline.add_component("joiner", DocumentJoiner(join_mode="concatenate"))
        pipeline.add_component("ranker", TransformersSimilarityRanker(model="BAAI/bge-reranker-v2-m3", top_k=5))

        # ---------------------------------------------------------
        # SYNCHRONIZED PROMPT (Identical to Ollama Script)
        # ---------------------------------------------------------
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
"""

        chat_template = [ChatMessage.from_user(template)]
        pipeline.add_component(
            "chat_prompt_builder",
            ChatPromptBuilder(template=chat_template, required_variables=["documents", "query"]),
        )

        pipeline.connect("query_embedder.embedding", "embedding_retriever.query_embedding")
        pipeline.connect("bm25_retriever", "joiner")
        pipeline.connect("embedding_retriever", "joiner")
        pipeline.connect("joiner.documents", "ranker.documents")
        pipeline.connect("ranker", "chat_prompt_builder.documents")

        return pipeline

    def _generate_from_prompt(self, prompt_content: str) -> str:
        """
        Generates text using apply_chat_template to ensure the model follows instructions.
        """
        
        # 1. Format as a chat message for the tokenizer
        messages = [{"role": "user", "content": prompt_content}]
        
        # 2. Apply chat template (adds <|user|>, [INST], etc.)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 3. Set seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # 4. Generate with RAG-optimized parameters
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.1,  # Low temp for factual accuracy
                top_p=0.9,
                # REMOVED no_repeat_ngram_size (it destroys citations like [Doc 1])
                repetition_penalty=1.1,  # Mild penalty to stop infinite loops
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 5. Decode ONLY the new tokens (answer)
        # We slice output_ids to exclude the input prompt length
        new_tokens = output_ids[0][input_ids.shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return answer

    def query(self, pipeline: Pipeline, query: str, top_k: int = 5):
        start = time.time()
        candidate_k = top_k * 2

        # Run Retrieval
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

        retrieval_latency = time.time() - start

        # Get the constructed prompt content
        prompt_messages = result["chat_prompt_builder"]["prompt"]
        if isinstance(prompt_messages, list) and len(prompt_messages) > 0:
            msg = prompt_messages[0]
            full_prompt = msg.content if hasattr(msg, "content") else str(msg)
        else:
            full_prompt = str(prompt_messages)

        # Run Generation
        gen_start = time.time()
        answer = self._generate_from_prompt(full_prompt)
        total_latency = (time.time() - gen_start) + retrieval_latency

        # Get documents
        docs = result["ranker"]["documents"]
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                "index": i,
                "title": doc.meta.get("title", doc.meta.get("filename", "Unknown")),
                "filename": doc.meta.get("filename", "Unknown"),
                "score": getattr(doc, "score", doc.meta.get("score", 0.0)),
            })

        # -------------------------------------------------------
        # METRICS CALCULATION (Exact match to Ollama script)
        # -------------------------------------------------------
        
        # 1. GMKR Analysis
        uses_general_knowledge = False
        gmkr = 0.0
        rag_tokens = 0
        gk_tokens = 0
        
        # Check for the tag (case insensitive)
        if "[General Knowledge / Non-Guideline]" in answer or "[General Knowledge]" in answer:
            uses_general_knowledge = True
            # Normalize split marker
            clean_answer = answer.replace("[General Knowledge]", "[General Knowledge / Non-Guideline]")
            parts = clean_answer.split("[General Knowledge / Non-Guideline]", 1)
            
            if len(parts) == 2:
                rag_part = parts[0]
                gk_part = parts[1]
                rag_tokens = len(rag_part.split())
                gk_tokens = len(gk_part.split())
                total_tokens = rag_tokens + gk_tokens
                if total_tokens > 0:
                    gmkr = gk_tokens / total_tokens

        # 2. Retrieval Alignment Score
        retrieval_alignment = calculate_retrieval_alignment(answer, docs, top_n=3)

        # 3. RUR and Evidence Density
        # Strictly look in the **Citations** section first, then fallback
        citations_match = re.search(r'\*\*Citations\*\*\s*(.+?)(?:\n\n|\Z)', answer, re.IGNORECASE | re.DOTALL)
        
        if citations_match:
            citations_text = citations_match.group(1)
            doc_numbers = re.findall(r'Doc\s*(\d+)', citations_text)
        else:
            # Fallback: find any [Doc X] in the whole text
            doc_numbers = re.findall(r'\[Doc\s*(\d+)\]', answer)
            
        n_distinct_docs = len(set(doc_numbers))
        n_citations_raw = len(doc_numbers)
        rur_flag = 1 if n_distinct_docs > 0 else 0
        
        answer_tokens_count = len(answer.split()) or 1
        evidence_density = (n_citations_raw / answer_tokens_count) * 100

        return {
            "answer": answer,
            "context_chunks": docs,
            "sources": sources,
            "latency_seconds": total_latency,
            "success": True,
            "uses_general_knowledge": uses_general_knowledge,
            "gmkr": round(gmkr, 4),
            "rag_tokens": rag_tokens,
            "gk_tokens": gk_tokens,
            "retrieval_alignment": round(retrieval_alignment, 4),
            "rur_flag": rur_flag,
            "n_distinct_docs": n_distinct_docs,
            "n_citations_raw": n_citations_raw,
            "n_tokens": answer_tokens_count,
            "evidence_density": round(evidence_density, 2),
        }

def load_all_prompts(simple_path="simple_prompts.json", complex_path="complex_prompts.json"):
    """Load prompts (same logic as Ollama script)"""
    base_dir = Path(__file__).resolve().parent
    
    with open(base_dir / simple_path, "r") as f:
        simple = json.load(f)
    with open(base_dir / complex_path, "r") as f:
        complex_ = json.load(f)

    prompts = []
    for p in simple + complex_:
        prompts.append({
            "id": p.get("id"),
            "description": p.get("title", p.get("id", "")),
            "prompt": p["prompt"],
            "expected_format": p.get("expected_format", []),
        })
    
    prompts.sort(key=lambda x: x["id"] or "")
    return prompts

def benchmark_model(rag_system: HFRAGSystem, prompts: list):
    print(f"\n🤖 HF Model: {rag_system.model_id} (key={rag_system.model_key})")
    
    pipeline = rag_system.create_hybrid_pipeline()
    results = []
    
    # OUTPUT DIRECTORY
    output_dir = Path("results_hf_SAPBERT_RAG")
    output_dir.mkdir(exist_ok=True)
    
    safe_model_name = rag_system.model_id.replace(":", "_").replace("/", "_")
    output_txt = output_dir / f"{safe_model_name}_results.txt"
    output_json = output_dir / f"{safe_model_name}_results.json"

    with open(output_txt, "w") as f:
        f.write(f"Model: {rag_system.model_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"   🔎 Query {i}/{len(prompts)} → {prompt['id']}")
        try:
            result = rag_system.query(pipeline, prompt["prompt"])
            
            # Print preview to console
            print(f"      Latency: {result['latency_seconds']:.2f}s | Citations: {result['n_citations_raw']}")

            result_entry = {
                "id": prompt.get("id"),
                "description": prompt["description"],
                "prompt": prompt["prompt"],
                "answer": result["answer"],
                "latency": result["latency_seconds"],
                "sources": result["sources"],
                # Metrics
                "uses_general_knowledge": result["uses_general_knowledge"],
                "gmkr": result["gmkr"],
                "retrieval_alignment": result["retrieval_alignment"],
                "rur_flag": result["rur_flag"],
                "n_distinct_docs": result["n_distinct_docs"],
                "n_citations_raw": result["n_citations_raw"],
                "evidence_density": result["evidence_density"],
            }
            results.append(result_entry)
            
            # Write to TXT immediately
            with open(output_txt, "a") as f:
                f.write(f"ID: {result_entry['id']}\n")
                f.write(f"Query: {result_entry['description']}\n")
                f.write(f"Latency: {result_entry['latency']:.2f}s\n")
                f.write(f"GMKR: {result_entry['gmkr']:.4f}\n")
                f.write(f"Retrieval Alignment: {result_entry['retrieval_alignment']:.4f}\n")
                f.write(f"Evidence Density: {result_entry['evidence_density']:.2f}\n")
                f.write(f"Answer:\n{result_entry['answer']}\n\n")
                f.write("Sources:\n")
                for src in result_entry["sources"]:
                    f.write(f"  [{src['index']}] {src['title']} ({src['score']:.3f})\n")
                f.write("\n" + "-" * 40 + "\n\n")

            # Update JSON immediately
            with open(output_json, "w") as jf:
                json.dump({
                    "model": rag_system.model_id,
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                    # Rolling averages
                    "summary": {
                        "avg_gmkr": sum(r["gmkr"] for r in results) / len(results),
                        "avg_ras": sum(r["retrieval_alignment"] for r in results) / len(results),
                        "avg_evidence_density": sum(r["evidence_density"] for r in results) / len(results)
                    }
                }, jf, indent=2)

        except Exception as e:
            logger.error(f"Error on prompt {prompt['id']}: {e}")
            continue

    print(f"\n✅ Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", type=str, required=True, choices=list(HF_MODEL_MAP.keys()))
    parser.add_argument("--cache", type=str, default="haystack_store/cached_documents_sapbert.pkl")
    args = parser.parse_args()

    print("\n🏥 HF RAG BENCHMARK (FIXED VERSION)\n")
    prompts = load_all_prompts()
    
    rag_system = HFRAGSystem(model_key=args.model_key)
    rag_system.load_documents_from_cache(args.cache)
    
    benchmark_model(rag_system, prompts)

if __name__ == "__main__":
    main()