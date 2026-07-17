"""
Configuration file for Transplant Infections RAG Benchmarking Platform

This file contains all the configuration settings, model specifications,
and system constants used throughout the application.

Purpose:
- Centralize all configuration settings
- Define model specifications for AWS Bedrock
- Set up evaluation parameters
- Configure file paths and directories
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

@dataclass
class SystemConfig:
    """Main system configuration settings"""
    
    # Project directories
    BASE_DIR: str = "."
    DATA_DIR: str = "./data"
    PAPERS_DIR: str = "./Papers"
    STORAGE_DIR: str = "./data/papers_storage"
    VECTOR_STORE_DIR: str = "./data/vector_store"
    RESULTS_DIR: str = "./results"
    LOGS_DIR: str = "./logs"
    
    # Database settings
    DATABASE_PATH: str = "./data/documents.db"
    
    # Processing settings (standardized)
    CHUNK_SIZE: int = 1500  # Characters (standardized from rag_trinf.py)
    CHUNK_OVERLAP: int = 100  # Characters
    MAX_FILE_SIZE_MB: int = 50
    BATCH_SIZE: int = 10
    
    # Vector store settings
    # EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    #EMBEDDING_DIMENSION: int = 384
    #EMBEDDING_MODEL: str = "BAAI/bge-m3"  # Much stronger medical understanding
    #EMBEDDING_DIMENSION: int = 1024       # Higher resolution vectors
    #VECTOR_SEARCH_TOP_K: int = 10

    # SapBERT config (for cached_documents_sapbert.pkl)
    EMBEDDING_MODEL: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    EMBEDDING_DIMENSION: int = 768
    VECTOR_SEARCH_TOP_K: int = 10
    
    # AWS settings
    AWS_REGION: str = "us-east-1"
    
    # API settings
    API_URL: str = "https://transplant-infections-app-production.up.railway.app/query/"
    LOCAL_API_URL: str = "http://localhost:8000/query/"
    API_TIMEOUT: int = 30
    
    # RAG settings
    DEFAULT_RAG_MODE: str = "auto"  # auto, openai, bedrock, hybrid
    TOP_K_CONTEXT: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Model settings
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_TEMPERATURE: float = 0.1
    
    # Evaluation settings
    SIGNIFICANCE_LEVEL: float = 0.05
    MIN_SAMPLE_SIZE: int = 30
    EXPERT_EVALUATION_BATCH_SIZE: int = 10

# =============================================================================
# AWS BEDROCK MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class BedrockModelConfig:
    """Configuration for AWS Bedrock models"""
    model_id: str
    model_name: str
    provider: str
    max_tokens: int
    temperature: float
    top_p: float
    pricing_per_1k_input: float
    pricing_per_1k_output: float
    context_window: int = 4096

class ModelConfigurations:
    """All available Bedrock model configurations"""
    
    MODELS = {
        # Anthropic Claude Models
        "claude-3-opus": BedrockModelConfig(
            model_id="anthropic.claude-3-opus-20240229-v1:0",
            model_name="Claude 3 Opus",
            provider="Anthropic",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=15.0,
            pricing_per_1k_output=75.0,
            context_window=200000
        ),
        "claude-3-sonnet": BedrockModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_name="Claude 3 Sonnet",
            provider="Anthropic",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=3.0,
            pricing_per_1k_output=15.0,
            context_window=200000
        ),
        "claude-3-haiku": BedrockModelConfig(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_name="Claude 3 Haiku",
            provider="Anthropic",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=0.25,
            pricing_per_1k_output=1.25,
            context_window=200000
        ),
        
        # Amazon Titan Models
        "titan-text-express": BedrockModelConfig(
            model_id="amazon.titan-text-express-v1",
            model_name="Titan Text Express",
            provider="Amazon",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=0.8,
            pricing_per_1k_output=1.6,
            context_window=8000
        ),
        
        # Meta Llama Models
        "llama2-70b": BedrockModelConfig(
            model_id="meta.llama2-70b-chat-v1",
            model_name="Llama 2 70B Chat",
            provider="Meta",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=1.95,
            pricing_per_1k_output=2.56,
            context_window=4096
        ),
        "llama3-70b": BedrockModelConfig(
            model_id="meta.llama3-70b-instruct-v1:0",
            model_name="Llama 3 70B Instruct",
            provider="Meta",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=2.65,
            pricing_per_1k_output=3.5,
            context_window=8192
        ),
        
        # Cohere Command Models
        "command-text": BedrockModelConfig(
            model_id="cohere.command-text-v14",
            model_name="Command Text",
            provider="Cohere",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=1.5,
            pricing_per_1k_output=2.0,
            context_window=4096
        ),
        
        # AI21 Jurassic Models
        "jurassic-2-ultra": BedrockModelConfig(
            model_id="ai21.j2-ultra-v1",
            model_name="Jurassic-2 Ultra",
            provider="AI21",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=15.0,
            pricing_per_1k_output=15.0,
            context_window=8192
        ),
        
        # Mistral AI Models
        "mistral-7b": BedrockModelConfig(
            model_id="mistral.mistral-7b-instruct-v0:2",
            model_name="Mistral 7B Instruct",
            provider="Mistral AI",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=0.15,
            pricing_per_1k_output=0.2,
            context_window=8192
        ),
        "mixtral-8x7b": BedrockModelConfig(
            model_id="mistral.mixtral-8x7b-instruct-v0:1",
            model_name="Mixtral 8x7B Instruct",
            provider="Mistral AI",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            pricing_per_1k_input=0.45,
            pricing_per_1k_output=0.7,
            context_window=32768
        )
    }

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

class EvaluationConfig:
    """Configuration for evaluation metrics and parameters"""
    
    # Automated metrics to calculate
    AUTOMATED_METRICS = [
        "bleu_score",
        "rouge_l", 
        "bertscore_f1",
        "semantic_similarity",
        "medical_entity_coverage",
        "terminology_accuracy",
        "citation_precision",
        "citation_recall",
        "factual_consistency",
        "retrieval_precision",
        "retrieval_recall",
        "context_relevance",
        "answer_relevance",
        "faithfulness_score"
    ]
    
    # Expert evaluation metrics (1-5 Likert scale)
    EXPERT_METRICS = [
        "clinical_accuracy",
        "clinical_relevance", 
        "safety_assessment",
        "completeness",
        "clarity",
        "actionability"
    ]
    
    # Statistical analysis parameters
    STATISTICAL_CONFIG = {
        "significance_level": 0.05,
        "confidence_interval": 0.95,
        "effect_size_threshold": 0.5,
        "min_sample_size": 30,
        "bootstrap_iterations": 1000
    }
    
    # Models to evaluate (subset of available models)
    MODELS_TO_EVALUATE = [
        "claude-3-opus",
        "claude-3-sonnet", 
        "claude-3-haiku",
        "llama3-70b",
        "llama2-70b",
        "mistral-7b",
        "mixtral-8x7b",
        "titan-text-express"
    ]

# =============================================================================
# MEDICAL DOMAIN CONFIGURATION
# =============================================================================

class MedicalConfig:
    """Medical domain specific configurations"""
    
    # Specialty areas for document tagging
    SPECIALTY_AREAS = {
        'transplant': ['transplant', 'graft', 'donor', 'recipient', 'immunosuppression'],
        'infection': ['infection', 'bacterial', 'viral', 'fungal', 'antimicrobial', 'antibiotic'],
        'immunology': ['immune', 'immunosuppressive', 'rejection', 'tolerance'],
        'nephrology': ['kidney', 'renal', 'dialysis'],
        'hepatology': ['liver', 'hepatic'],
        'cardiology': ['heart', 'cardiac', 'cardiovascular'],
        'pulmonology': ['lung', 'pulmonary', 'respiratory']
    }
    
    # Document types
    DOCUMENT_TYPES = [
        "research_paper",
        "review", 
        "case_study",
        "guideline",
        "systematic_review",
        "meta_analysis"
    ]
    
    # Quality assessment criteria
    QUALITY_CRITERIA = {
        "content_length_weight": 0.2,
        "section_structure_weight": 0.3,
        "metadata_completeness_weight": 0.2,
        "medical_relevance_weight": 0.3
    }

# =============================================================================
# FILE PATHS AND TEMPLATES
# =============================================================================

class PathConfig:
    """File path configurations"""
    
    @staticmethod
    def create_directories():
        """Create all necessary directories"""
        config = SystemConfig()
        directories = [
            config.DATA_DIR,
            config.STORAGE_DIR,
            config.VECTOR_STORE_DIR,
            config.RESULTS_DIR,
            config.LOGS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ All directories created successfully")

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class PromptTemplates:
    """Template prompts for different models and use cases"""
    
    MEDICAL_RAG_PROMPT = """You are a medical AI assistant specializing in transplant infections. 
Use the provided research documents to answer the clinical query accurately and comprehensively.

Context Documents:
{context}

Query: {query}

Instructions:
1. Base your response primarily on the provided documents
2. Cite specific documents when making claims (use "Document X")
3. If information is not in the documents, clearly state this
4. Provide clinically actionable information when appropriate
5. Maintain medical accuracy and safety
6. Use appropriate medical terminology

Response:"""

    EXPERT_EVALUATION_PROMPT = """Please evaluate this AI response for a clinical query about transplant infections.

Query: {query}
AI Response: {response}
Reference Documents: {context}

Evaluation Criteria:
1. Clinical Accuracy (1-5): How factually correct is the medical information?
2. Clinical Relevance (1-5): How relevant is the response to the query?
3. Safety Assessment (1-5): How safe are the recommendations?
4. Completeness (1-5): How complete is the response?
5. Clarity (1-5): How clear and understandable is the response?
6. Actionability (1-5): How actionable are the recommendations?

Please provide ratings and detailed comments."""

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Set up the environment and create necessary directories"""
    
    # Create directories
    PathConfig.create_directories()
    
    # Set environment variables if not already set
    if not os.getenv("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = SystemConfig().AWS_REGION
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{SystemConfig().LOGS_DIR}/app.log"),
            logging.StreamHandler()
        ]
    )
    
    print("✅ Environment setup completed")

# Initialize configuration on import
if __name__ == "__main__":
    setup_environment()