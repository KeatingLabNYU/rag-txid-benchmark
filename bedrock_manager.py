"""
AWS Bedrock Manager
================================================================================
Manages AWS Bedrock model configurations and API calls for the RAG benchmark.

Key implementation notes:
- Llama 3 legacy models: max_tokens capped at 2048 to avoid AWS ValidationException
- GPT-OSS reasoning models: strip reasoningContent blocks; return only final text block
- Amazon Nova and Cohere models: see run_bedrock_model_with_RAG_metrics.py for
  provider-specific inference-config patches
"""

import os
import time
import logging
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_text_from_bedrock_content_block(block: Any) -> str:
    if isinstance(block, str):
        return block.strip()

    if not isinstance(block, dict):
        return ""

    # Reasoning blocks are returned separately for GPT-OSS and should not be used
    # as the user-facing answer when a final text block is also present.
    if "reasoningContent" in block:
        return ""

    text = block.get("text")
    if isinstance(text, str):
        raw_text = text.strip()
        if not raw_text:
            return ""
        if raw_text.startswith("{") or raw_text.startswith("["):
            try:
                parsed = json.loads(raw_text)
                extracted = _extract_text_from_bedrock_content_block(parsed)
                if extracted:
                    return extracted
            except Exception:
                pass
        return raw_text

    content = block.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [_extract_text_from_bedrock_content_block(item) for item in content]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        return _extract_text_from_bedrock_content_block(content)

    return ""

@dataclass
class ModelConfig:
    model_id: str
    model_name: str
    provider: str
    max_tokens: int = 4096  # Default for most models
    temperature: float = 0.1
    top_p: float = 0.9

class BedrockModels:
    """All Bedrock model configs used in the benchmark."""

    # --- AI21 ---
    JAMBA_1_5_LARGE = ModelConfig(
        model_id="ai21.jamba-1-5-large-v1:0",
        model_name="Jamba 1.5 Large",
        provider="AI21 Labs",
        max_tokens=4096
    )

    # --- OpenAI (reasoning models) ---
    GPT_OSS_120B = ModelConfig(
        model_id="openai.gpt-oss-120b-1:0",
        model_name="GPT-OSS 120B",
        provider="OpenAI"
    )
    
    GPT_OSS_20B = ModelConfig(
        model_id="openai.gpt-oss-20b-1:0",
        model_name="GPT-OSS 20B",
        provider="OpenAI"
    )

    # --- Meta Llama (legacy native API; max_tokens capped at 2048) ---
    LLAMA_3_70B = ModelConfig(
        model_id="meta.llama3-70b-instruct-v1:0",
        model_name="Llama 3 70B (Legacy)",
        provider="Meta",
        max_tokens=2048 
    )
    
    LLAMA_3_8B = ModelConfig(
        model_id="meta.llama3-8b-instruct-v1:0",
        model_name="Llama 3 8B (Legacy)",
        provider="Meta",
        max_tokens=2048
    )

    # --- Anthropic Claude (legacy native API) ---
    CLAUDE_3_SONNET = ModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_name="Claude 3 Sonnet",
        provider="Anthropic"
    )
    
    CLAUDE_3_HAIKU = ModelConfig(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_name="Claude 3 Haiku",
        provider="Anthropic"
    )

    # --- Mistral & Amazon Nova ---
    MISTRAL_LARGE = ModelConfig(
        model_id="mistral.mistral-large-2402-v1:0",
        model_name="Mistral Large (2402)",
        provider="Mistral AI",
        max_tokens=8192
    )
    
    NOVA_PRO = ModelConfig(
        model_id="amazon.nova-pro-v1:0",
        model_name="Nova Pro",
        provider="Amazon"
    )

    NOVA_LITE = ModelConfig(
        model_id="amazon.nova-lite-v1:0",
        model_name="Nova Lite (v1)",
        provider="Amazon"
    )

    NOVA_MICRO = ModelConfig(
        model_id="amazon.nova-micro-v1:0",
        model_name="Nova Micro",
        provider="Amazon"
    )

    @classmethod
    def get_all_models(cls) -> List[ModelConfig]:
        return [
            cls.JAMBA_1_5_LARGE,
            cls.GPT_OSS_120B,
            cls.GPT_OSS_20B,
            cls.LLAMA_3_70B,
            cls.LLAMA_3_8B,
            cls.CLAUDE_3_SONNET,
            cls.CLAUDE_3_HAIKU,
            cls.MISTRAL_LARGE,
            cls.NOVA_PRO,
            cls.NOVA_LITE,
            cls.NOVA_MICRO
        ]

class BedrockManager:
    def __init__(self, region: str = "us-east-1", profile_name: str = "default"):
        """Initialize Bedrock Manager.

        Args:
            region: AWS region (default: us-east-1)
            profile_name: AWS CLI profile name (default: "default"). Pass None to
                          use environment credentials (IAM role, env vars, etc.).
        """
        self.region = region
        self.profile_name = profile_name
        self.bedrock_client = None
        self._initialize_bedrock_client()

    def _initialize_bedrock_client(self):
        try:
            config = Config(read_timeout=180, retries={"max_attempts": 3, "mode": "adaptive"})
            session = boto3.Session(profile_name=self.profile_name)
            self.bedrock_client = session.client(
                service_name="bedrock-runtime",
                region_name=self.region,
                config=config,
            )
            logger.info("✅ Bedrock client initialized")
        except Exception as e:
            logger.error(f"⚠️  Bedrock init failed: {e}")

    def query_model(self, model_config: ModelConfig, query: str, context: str, 
                   system_prompt: str = "You are a helpful expert.") -> Dict[str, Any]:
        
        start_time = time.time()
        if not self.bedrock_client: return {'success': False, 'error': "Client not initialized"}

        messages = [{"role": "user", "content": [{"text": f"Context:\n{context}\n\nQuestion: {query}"}]}]
        system = [{"text": system_prompt}]

        try:
            response = self.bedrock_client.converse(
                modelId=model_config.model_id,
                messages=messages,
                system=system,
                inferenceConfig={"maxTokens": model_config.max_tokens, "temperature": model_config.temperature}
            )
            
            # --- IMPROVED PARSER (Handles Reasoning Models) ---
            output = response.get('output', {})
            message = output.get('message', {})
            content_list = message.get('content', [])

            answer_parts = [_extract_text_from_bedrock_content_block(item) for item in content_list]
            answer = "\n".join(part for part in answer_parts if part).strip()

            if not answer and content_list:
                answer = str(content_list[0])

            return {
                'model_name': model_config.model_name,
                'answer': answer,
                'latency_seconds': time.time() - start_time,
                'success': True
            }

        except ClientError as e:
            err = e.response['Error']
            return {
                'model_name': model_config.model_name,
                'success': False,
                'error': f"{err.get('Code')}: {err.get('Message')}",
                'latency_seconds': time.time() - start_time
            }
        except Exception as e:
            return {
                'model_name': model_config.model_name,
                'success': False,
                'error': str(e),
                'latency_seconds': time.time() - start_time
            }