"""
DOCUMENT PROCESSING ENGINE
================================================================================

This module provides advanced document processing capabilities specifically
designed for medical research papers, particularly in the transplant infections domain.

Key Features:
1. Advanced PDF text extraction with multiple fallback methods
2. Medical entity recognition using ScispaCy
3. Intelligent section identification (Abstract, Methods, Results, etc.)
4. Text chunking with semantic awareness
5. Embedding generation for vector similarity search
6. Medical relevance and quality scoring
7. Citation and reference extraction

The processing pipeline is designed to handle the complexities of medical
literature while maintaining high accuracy and performance.

Components:
- DocumentTextProcessor: Main text processing and chunking
- MetadataExtractor: Bibliographic information extraction
- DocumentQualityAssessor: Quality scoring for medical relevance
- MedicalEntityExtractor: Medical concept identification

Dependencies:
- PyPDF2, pdfplumber, PyMuPDF for PDF processing
- spaCy and scispaCy for medical NLP
- sentence-transformers for embeddings
- NLTK for text processing utilities

Usage:
    from document_processor import DocumentTextProcessor
    
    processor = DocumentTextProcessor()
    result = processor.process_document("/path/to/paper.pdf")
    
    # Access processed content
    chunks = result['chunks']
    sections = result['sections']
    embeddings = result['embeddings']
"""

import os
import re
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime

# Core text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# PDF processing with multiple fallback methods
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF

# NLP and embeddings
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not available - using basic text processing")

try:
    import scispacy
    import spacy
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False
    print("⚠️  scispaCy not available - using basic medical entity extraction")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available - embeddings will be disabled")

# OCR support
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    Image = None  # type: ignore
    print("⚠️  pytesseract/Pillow not available - OCR fallback disabled")

# Configuration
from config import SystemConfig, MedicalConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# TEXT PREPROCESSING UTILITIES
# =============================================================================

class TextPreprocessor:
    """Utility class for text preprocessing and cleaning."""
    
    def __init__(self):
        """Initialize with NLTK downloads."""
        self._ensure_nltk_data()
        
        # Compile regex patterns for efficiency
        self.patterns = {
            'whitespace': re.compile(r'\s+'),
            'line_breaks': re.compile(r'\n+'),
            'page_breaks': re.compile(r'\f+'),
            'figure_refs': re.compile(r'(?i)\b(?:figure|fig\.?|table|tab\.?)\s*\d+\b'),
            'citations': re.compile(r'\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'doi': re.compile(r'(?i)doi:\s*(10\.\d+/[^\s]+)'),
            'pmid': re.compile(r'(?i)pmid:\s*(\d+)'),
            'section_headers': re.compile(r'(?i)^(abstract|introduction|background|methods?|methodology|materials?\s+and\s+methods?|results?|discussion|conclusions?|references?|acknowledgments?)\s*:?\s*$', re.MULTILINE)
        }
    
    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove page breaks and excessive whitespace
        text = self.patterns['page_breaks'].sub(' ', text)
        text = self.patterns['line_breaks'].sub('\n', text)
        text = self.patterns['whitespace'].sub(' ', text)
        
        # Remove URLs and emails (often metadata/headers)
        text = self.patterns['urls'].sub('', text)
        text = self.patterns['email'].sub('', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{2,}', '-', text)
        
        return text.strip()
    
    def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata elements from text content.
        
        Args:
            text: Text to extract metadata from
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Extract DOI
        doi_match = self.patterns['doi'].search(text)
        if doi_match:
            metadata['doi'] = doi_match.group(1)
        
        # Extract PMID
        pmid_match = self.patterns['pmid'].search(text)
        if pmid_match:
            metadata['pmid'] = pmid_match.group(1)
        
        # Count citations
        citations = self.patterns['citations'].findall(text)
        metadata['citation_count'] = len(citations)
        
        # Count figure/table references
        figure_refs = self.patterns['figure_refs'].findall(text)
        metadata['figure_table_count'] = len(figure_refs)
        
        return metadata

# =============================================================================
# PDF PROCESSING WITH MULTIPLE METHODS
# =============================================================================

class PDFProcessor:
    """Advanced PDF processing with multiple extraction methods and fallbacks."""
    
    def __init__(self):
        """Initialize PDF processor with method preferences."""
        self.text_preprocessor = TextPreprocessor()
        
        # Method preferences (in order of preference)
        self.extraction_methods = [
            ('pdfplumber', self._extract_with_pdfplumber),
            ('pymupdf', self._extract_with_pymupdf),
            ('pypdf2', self._extract_with_pypdf2)
        ]

        if OCR_AVAILABLE:
            self.extraction_methods.append(('ocr', self._extract_with_ocr))
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using multiple methods with fallbacks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Try each extraction method
        for method_name, method_func in self.extraction_methods:
            try:
                logger.debug(f"Trying PDF extraction with {method_name}...")
                result = method_func(str(pdf_path))
                
                if result and result.get('text', '').strip():
                    logger.debug(f"✅ PDF extraction successful with {method_name}")
                    result['extraction_method'] = method_name
                    return result
                else:
                    logger.debug(f"❌ {method_name} returned empty text")
                    
            except Exception as e:
                logger.debug(f"❌ {method_name} failed: {str(e)}")
                continue
        
        # If all methods fail, return basic structure
        logger.warning(f"All PDF extraction methods failed for {pdf_path}")
        return {
            'text': f"[Failed to extract text from {pdf_path.name}]",
            'pages': 0,
            'extraction_method': 'failed',
            'metadata': {}
        }

    def _pixmap_to_image(self, pixmap: "fitz.Pixmap"):
        """Convert a PyMuPDF pixmap to a Pillow image."""
        # PyMuPDF pixmaps may be grayscale (1 component) or RGB(A)
        if pixmap.alpha:
            mode = "RGBA"
        else:
            mode = "RGB"
        image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
        return image

    def _extract_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using OCR via PyMuPDF rendering and pytesseract."""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR not available")

        doc = fitz.open(pdf_path)
        pages_text: List[str] = []

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                # Render page as image at higher DPI for better OCR accuracy
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = self._pixmap_to_image(pix)
                ocr_text = pytesseract.image_to_string(image)
                cleaned_text = self.text_preprocessor.clean_text(ocr_text)
                if cleaned_text:
                    pages_text.append(cleaned_text)
            except Exception as e:
                logger.debug(f"OCR failed on page {page_num}: {e}")
                continue

        page_count = doc.page_count
        doc.close()

        full_text = '\n\n'.join(pages_text)

        if not full_text.strip():
            logger.debug("OCR extraction produced no text")

        return {
            'text': full_text,
            'pages': page_count,
            'metadata': {'ocr': True, 'ocr_pages': len(pages_text)}
        }
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using pdfplumber (best for complex layouts)."""
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = self.text_preprocessor.clean_text(page_text)
                        pages_text.append(cleaned_text)
                except Exception as e:
                    logger.debug(f"Failed to extract page {page_num}: {e}")
                    continue
            
            full_text = '\n\n'.join(pages_text)
            
            # Extract basic metadata
            metadata = {}
            if pdf.metadata:
                metadata.update({
                    'title': str(pdf.metadata.get('Title', '')),
                    'author': str(pdf.metadata.get('Author', '')),
                    'creator': str(pdf.metadata.get('Creator', '')),
                    'subject': str(pdf.metadata.get('Subject', ''))
                })
            
            return {
                'text': full_text,
                'pages': len(pdf.pages),
                'metadata': metadata
            }
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF (good for performance)."""
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                if page_text:
                    cleaned_text = self.text_preprocessor.clean_text(page_text)
                    pages_text.append(cleaned_text)
            except Exception as e:
                logger.debug(f"Failed to extract page {page_num}: {e}")
                continue
        
        full_text = '\n\n'.join(pages_text)
        
        # Extract metadata
        metadata = {}
        if doc.metadata:
            metadata.update({
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'creator': doc.metadata.get('creator', ''),
                'subject': doc.metadata.get('subject', '')
            })
        
        doc.close()
        
        return {
            'text': full_text,
            'pages': doc.page_count,
            'metadata': metadata
        }
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyPDF2 (basic fallback)."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_text = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = self.text_preprocessor.clean_text(page_text)
                        pages_text.append(cleaned_text)
                except Exception as e:
                    logger.debug(f"Failed to extract page {page_num}: {e}")
                    continue
            
            full_text = '\n\n'.join(pages_text)
            
            # Extract metadata
            metadata = {}
            if pdf_reader.metadata:
                metadata.update({
                    'title': str(pdf_reader.metadata.get('/Title', '')),
                    'author': str(pdf_reader.metadata.get('/Author', '')),
                    'creator': str(pdf_reader.metadata.get('/Creator', '')),
                    'subject': str(pdf_reader.metadata.get('/Subject', ''))
                })
            
            return {
                'text': full_text,
                'pages': len(pdf_reader.pages),
                'metadata': metadata
            }

# =============================================================================
# SECTION IDENTIFICATION AND PARSING
# =============================================================================

class SectionParser:
    """Intelligent section identification for medical papers."""
    
    def __init__(self):
        """Initialize with medical paper section patterns."""
        self.section_patterns = {
            'abstract': [
                r'(?i)^(?:abstract|summary)(?:\s|:|\n)',
                r'(?i)\babstract\b.*?(?=\n\s*(?:key\s*words?|introduction|background|\d+\.|\w+:))',
            ],
            'keywords': [
                r'(?i)(?:key\s*words?|keywords?)[:.]?\s*(.*?)(?=\n\s*(?:introduction|background|\d+\.))',
            ],
            'introduction': [
                r'(?i)(?:^|\n\s*)(?:1\.?\s*)?(?:introduction|background)(?:\s|:|\n)',
            ],
            'methods': [
                r'(?i)(?:^|\n\s*)(?:\d+\.?\s*)?(?:methods?|methodology|materials?\s+and\s+methods?|experimental\s+(?:design|procedure))(?:\s|:|\n)',
            ],
            'results': [
                r'(?i)(?:^|\n\s*)(?:\d+\.?\s*)?(?:results?|findings)(?:\s|:|\n)',
            ],
            'discussion': [
                r'(?i)(?:^|\n\s*)(?:\d+\.?\s*)?(?:discussion)(?:\s|:|\n)',
            ],
            'conclusion': [
                r'(?i)(?:^|\n\s*)(?:\d+\.?\s*)?(?:conclusions?|concluding\s+remarks?)(?:\s|:|\n)',
            ],
            'references': [
                r'(?i)(?:^|\n\s*)(?:references?|bibliography|literature\s+cited)(?:\s|:|\n)',
            ],
            'acknowledgments': [
                r'(?i)(?:^|\n\s*)(?:acknowledgments?|acknowledgements?)(?:\s|:|\n)',
            ]
        }
    
    def parse_sections(self, text: str) -> Dict[str, str]:
        """
        Parse text into identified sections.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        remaining_text = text
        
        # Find section boundaries
        section_boundaries = []
        
        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
                for match in matches:
                    section_boundaries.append((match.start(), section_name, match.end()))
        
        # Sort boundaries by position
        section_boundaries.sort(key=lambda x: x[0])
        
        # Extract content between boundaries
        for i, (start_pos, section_name, header_end) in enumerate(section_boundaries):
            # Find end position (start of next section or end of text)
            if i + 1 < len(section_boundaries):
                end_pos = section_boundaries[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract section content
            section_content = text[header_end:end_pos].strip()
            
            # Clean and validate section content
            if section_content and len(section_content) > 10:
                sections[section_name] = section_content
        
        # If no sections found, try simpler patterns
        if not sections:
            sections = self._fallback_section_parsing(text)
        
        return sections
    
    def _fallback_section_parsing(self, text: str) -> Dict[str, str]:
        """Fallback section parsing for documents without clear structure."""
        sections = {}
        
        # Split by common patterns
        paragraphs = text.split('\n\n')
        
        # Look for abstract in first few paragraphs
        for i, para in enumerate(paragraphs[:3]):
            if len(para) > 100 and ('background' in para.lower() or 'objective' in para.lower()):
                sections['abstract'] = para
                break
        
        # If document is long enough, assume it has body content
        if len(text) > 1000:
            # Take middle portion as methods/results
            text_length = len(text)
            start_idx = text_length // 4
            end_idx = 3 * text_length // 4
            sections['body'] = text[start_idx:end_idx]
        else:
            # Short document - treat as single body
            sections['body'] = text
        
        return sections

# =============================================================================
# MEDICAL ENTITY EXTRACTION
# =============================================================================

class MedicalEntityExtractor:
    """Medical entity extraction using scispaCy and pattern matching."""
    
    def __init__(self):
        """Initialize medical entity extractor."""
        self.medical_config = MedicalConfig()
        self.nlp = None
        self._load_models()
        
        # Medical term patterns
        self.medical_patterns = {
            'medications': [
                r'\b(?:cyclosporine|tacrolimus|mycophenolate|azathioprine|prednisone|prednisolone|sirolimus|everolimus)\b',
                r'\b(?:immunosuppressive?|immunosuppressant)\b',
                r'\b(?:antibiotic|antiviral|antifungal|antimicrobial)\b',
            ],
            'organisms': [
                r'\b(?:CMV|cytomegalovirus|EBV|epstein-barr)\b',
                r'\b(?:candida|aspergillus|pneumocystis)\b',
                r'\b(?:bacteria|virus|fungi|parasite)\w*\b',
            ],
            'procedures': [
                r'\b(?:transplant|transplantation|graft)\w*\b',
                r'\b(?:biopsy|endoscopy|bronchoscopy)\b',
                r'\b(?:dialysis|plasmapheresis)\b',
            ],
            'organs': [
                r'\b(?:kidney|renal|liver|hepatic|heart|cardiac|lung|pulmonary)\b',
                r'\b(?:pancreas|islet|cornea|bone\s+marrow)\b',
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.medical_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _load_models(self):
        """Load spaCy models with fallbacks."""
        if SCISPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_sci_lg")
                logger.debug("✅ Loaded scispaCy large model")
                return
            except OSError:
                logger.debug("scispaCy large model not found, trying medium...")
                try:
                    self.nlp = spacy.load("en_core_sci_md")
                    logger.debug("✅ Loaded scispaCy medium model")
                    return
                except OSError:
                    logger.debug("scispaCy medium model not found, trying small...")
                    try:
                        self.nlp = spacy.load("en_core_sci_sm")
                        logger.debug("✅ Loaded scispaCy small model")
                        return
                    except OSError:
                        pass
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.debug("✅ Loaded standard spaCy model")
                return
            except OSError:
                pass
        
        logger.warning("⚠️  No spaCy models available - using pattern matching only")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary mapping entity types to lists of found entities
        """
        entities = {
            'medications': [],
            'organisms': [],
            'procedures': [],
            'organs': [],
            'diseases': [],
            'symptoms': [],
            'measurements': []
        }
        
        # Use spaCy if available
        if self.nlp:
            entities.update(self._extract_with_spacy(text))
        
        # Use pattern matching
        pattern_entities = self._extract_with_patterns(text)
        
        # Merge results
        for category, values in pattern_entities.items():
            if category in entities:
                entities[category].extend(values)
        
        # Remove duplicates and clean
        for category in entities:
            entities[category] = list(set(
                entity.strip().lower() for entity in entities[category]
                if entity.strip() and len(entity.strip()) > 2
            ))
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER."""
        entities = {
            'diseases': [],
            'symptoms': [],
            'measurements': [],
            'medications': [],
            'procedures': []
        }
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_text = ent.text.strip()
                
                # Map spaCy labels to our categories
                if ent.label_ in ['DISEASE', 'SYMPTOM']:
                    entities['diseases'].append(entity_text)
                elif ent.label_ in ['CHEMICAL', 'DRUG']:
                    entities['medications'].append(entity_text)
                elif ent.label_ in ['PROCEDURE']:
                    entities['procedures'].append(entity_text)
                elif ent.label_ in ['MEASUREMENT', 'QUANTITY']:
                    entities['measurements'].append(entity_text)
                elif ent.label_ in ['SYMPTOM']:
                    entities['symptoms'].append(entity_text)
                    
        except Exception as e:
            logger.debug(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    def _extract_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns."""
        entities = {category: [] for category in self.compiled_patterns}
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                entities[category].extend(matches)
        
        return entities

# =============================================================================
# TEXT CHUNKING WITH SEMANTIC AWARENESS
# =============================================================================

class SemanticChunker:
    """Intelligent text chunking with semantic awareness."""
    
    def __init__(self, config: SystemConfig):
        """Initialize chunker with configuration."""
        self.config = config
        self.text_preprocessor = TextPreprocessor()
        
        # Initialize sentence tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def create_chunks(self, 
                     sections: Dict[str, str], 
                     chunk_size: Optional[int] = None,
                     overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from document sections.
        
        Args:
            sections: Dictionary of section name to content
            chunk_size: Maximum chunk size in tokens (uses config default if None)
            overlap: Overlap between chunks in tokens (uses config default if None)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        
        chunks = []
        global_chunk_index = 0
        
        # Process each section
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            
            section_chunks = self._chunk_section(
                section_text, section_name, chunk_size, overlap, global_chunk_index
            )
            
            chunks.extend(section_chunks)
            global_chunk_index += len(section_chunks)
        
        return chunks
    
    def _chunk_section(self, 
                      text: str, 
                      section_name: str, 
                      chunk_size: int,
                      overlap: int,
                      start_index: int) -> List[Dict[str, Any]]:
        """Chunk a single section with semantic awareness."""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        
        for sentence in sentences:
            # Estimate token count (rough approximation: word count * 1.3)
            sentence_tokens = len(word_tokenize(sentence))
            
            # Check if adding this sentence would exceed chunk size
            if (current_token_count + sentence_tokens > chunk_size and 
                current_chunk_sentences):
                
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunk_info = self._create_chunk_info(
                    chunk_text, section_name, len(chunks) + start_index
                )
                chunks.append(chunk_info)
                
                # Handle overlap
                if overlap > 0 and len(current_chunk_sentences) > 1:
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk_sentences[-max(1, overlap//10):]
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_token_count = sum(len(word_tokenize(s)) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = [sentence]
                    current_token_count = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_info = self._create_chunk_info(
                chunk_text, section_name, len(chunks) + start_index
            )
            chunks.append(chunk_info)
        
        return chunks
    
    def _create_chunk_info(self, text: str, section_type: str, chunk_index: int) -> Dict[str, Any]:
        """Create chunk information dictionary."""
        # Basic text statistics
        word_count = len(word_tokenize(text))
        sentence_count = len(sent_tokenize(text))
        
        # Calculate chunk quality score
        quality_score = self._calculate_chunk_quality(text, section_type, word_count)
        
        return {
            'text': text,
            'section_type': section_type,
            'chunk_index': chunk_index,
            'word_count': word_count,
            'char_count': len(text),
            'sentence_count': sentence_count,
            'quality_score': quality_score,
            'medical_entities': {},  # Will be filled by entity extractor
            'embedding': None        # Will be filled by embedding generator
        }
    
    def _calculate_chunk_quality(self, text: str, section_type: str, word_count: int) -> float:
        """Calculate quality score for a text chunk."""
        score = 0.0
        
        # Length appropriateness (0.3 weight)
        if 50 <= word_count <= 600:
            score += 0.3
        elif word_count >= 25:
            score += 0.15
        
        # Section importance (0.2 weight)
        section_weights = {
            'abstract': 0.2,
            'introduction': 0.15,
            'methods': 0.15,
            'results': 0.2,
            'discussion': 0.2,
            'conclusion': 0.15,
            'body': 0.1
        }
        score += section_weights.get(section_type, 0.1)
        
        # Content density (0.3 weight)
        # Check for medical terminology density
        medical_terms = 0
        text_lower = text.lower()
        for specialty, keywords in MedicalConfig.SPECIALTY_AREAS.items():
            medical_terms += sum(1 for keyword in keywords if keyword in text_lower)
        
        medical_density = min(medical_terms / word_count * 100, 1.0) if word_count > 0 else 0
        score += medical_density * 0.3
        
        # Readability (0.2 weight)
        sentences = sent_tokenize(text)
        if sentences:
            avg_sentence_length = word_count / len(sentences)
            if 10 <= avg_sentence_length <= 25:
                score += 0.2
            elif 5 <= avg_sentence_length <= 35:
                score += 0.1
        
        return min(score, 1.0)

# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

class EmbeddingGenerator:
    """Generate embeddings for text chunks using sentence transformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedding generator."""
        self.model_name = model_name or SystemConfig().EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️  sentence-transformers not available - embeddings disabled")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.debug(f"✅ Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load embedding model: {e}")
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Updated chunks with embeddings
        """
        if not self.model:
            logger.debug("Embedding model not available, skipping embeddings")
            return chunks
        
        try:
            # Extract texts for batch processing
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            logger.debug(f"✅ Generated embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate embeddings: {e}")
        
        return chunks

# =============================================================================
# MAIN DOCUMENT PROCESSING CLASSES
# =============================================================================

class DocumentTextProcessor:
    """Main document text processor combining all processing components."""
    
    def __init__(self):
        """Initialize all processing components."""
        self.config = SystemConfig()
        self.medical_config = MedicalConfig()
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.section_parser = SectionParser()
        self.entity_extractor = MedicalEntityExtractor()
        self.chunker = SemanticChunker(self.config)
        self.embedding_generator = EmbeddingGenerator()
        
        logger.debug("✅ DocumentTextProcessor initialized")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with processed document information
        """
        file_path = Path(file_path)
        logger.debug(f"🔄 Processing document: {file_path.name}")
        
        try:
            # Step 1: Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                extraction_result = self.pdf_processor.extract_text(str(file_path))
                full_text = extraction_result['text']
                extraction_metadata = extraction_result.get('metadata', {})
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text = f.read()
                extraction_metadata = {}
            
            if not full_text.strip():
                raise ValueError("No text content extracted from document")
            
            # Step 2: Parse sections
            logger.debug("📑 Parsing document sections...")
            sections = self.section_parser.parse_sections(full_text)
            
            # Step 3: Create chunks
            logger.debug("✂️  Creating text chunks...")
            chunks = self.chunker.create_chunks(sections)
            
            # Step 4: Extract medical entities for each chunk
            logger.debug("🏥 Extracting medical entities...")
            for chunk in chunks:
                entities = self.entity_extractor.extract_entities(chunk['text'])
                chunk['medical_entities'] = entities
            
            # Step 5: Generate embeddings
            logger.debug("🔗 Generating embeddings...")
            chunks = self.embedding_generator.generate_embeddings(chunks)
            
            # Step 6: Compile results
            result = {
                'full_text': full_text,
                'sections': sections,
                'chunks': chunks,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'section_count': len(sections),
                'chunk_count': len(chunks),
                'extraction_metadata': extraction_metadata,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_success': True
            }
            
            logger.debug(f"✅ Document processing complete: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            
            # Return error result
            return {
                'full_text': '',
                'sections': {},
                'chunks': [],
                'word_count': 0,
                'char_count': 0,
                'section_count': 0,
                'chunk_count': 0,
                'extraction_metadata': {},
                'processing_timestamp': datetime.now().isoformat(),
                'processing_success': False,
                'error': str(e)
            }

class MetadataExtractor:
    """Extract and enrich document metadata."""
    
    def __init__(self):
        """Initialize metadata extractor."""
        self.medical_config = MedicalConfig()
        self.text_preprocessor = TextPreprocessor()
    
    def extract(self, file_path: str, provided_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from document.
        
        Args:
            file_path: Path to document file
            provided_metadata: Optional metadata provided by user
            
        Returns:
            Dictionary with extracted and enriched metadata
        """
        file_path = Path(file_path)
        metadata = provided_metadata.copy() if provided_metadata else {}
        
        try:
            # Extract from file content
            if file_path.suffix.lower() == '.pdf':
                content_metadata = self._extract_from_pdf(str(file_path))
            else:
                content_metadata = self._extract_from_text_file(str(file_path))
            
            # Merge with provided metadata (provided takes precedence)
            for key, value in content_metadata.items():
                if key not in metadata or not metadata[key]:
                    metadata[key] = value
            
            # Extract from file path
            path_metadata = self._extract_from_path(file_path)
            for key, value in path_metadata.items():
                if key not in metadata or not metadata[key]:
                    metadata[key] = value
            
            # Set defaults and clean up
            metadata = self._finalize_metadata(metadata, file_path)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return self._create_fallback_metadata(file_path, provided_metadata)
    
    def _extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        metadata = {}
        
        try:
            # Try PyPDF2 first for metadata
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    metadata.update({
                        'title': str(pdf_meta.get('/Title', '')).strip(),
                        'authors': [str(pdf_meta.get('/Author', '')).strip()] if pdf_meta.get('/Author') else [],
                        'creator': str(pdf_meta.get('/Creator', '')).strip(),
                        'subject': str(pdf_meta.get('/Subject', '')).strip(),
                        'creation_date': str(pdf_meta.get('/CreationDate', '')).strip()
                    })
                
                # Extract text from first page for additional metadata
                if pdf_reader.pages:
                    first_page_text = pdf_reader.pages[0].extract_text()
                    text_metadata = self.text_preprocessor.extract_metadata_from_text(first_page_text)
                    metadata.update(text_metadata)
        
        except Exception as e:
            logger.debug(f"PDF metadata extraction failed: {e}")
        
        return metadata
    
    def _extract_from_text_file(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from text file."""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 2000 characters for metadata extraction
                text_sample = f.read(2000)
                
            text_metadata = self.text_preprocessor.extract_metadata_from_text(text_sample)
            metadata.update(text_metadata)
            
        except Exception as e:
            logger.debug(f"Text file metadata extraction failed: {e}")
        
        return metadata
    
    def _extract_from_path(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file path and name."""
        metadata = {}
        
        # Extract title from filename if not already set
        if not metadata.get('title'):
            title = file_path.stem
            # Clean up filename
            title = title.replace('_', ' ').replace('-', ' ')
            title = re.sub(r'\s+', ' ', title).strip()
            title = title.title()
            metadata['title'] = title
        
        # Determine document type from path
        path_lower = str(file_path).lower()
        if 'review' in path_lower:
            doc_type = 'review'
        elif 'case' in path_lower:
            doc_type = 'case_study'
        elif 'guideline' in path_lower:
            doc_type = 'guideline'
        elif 'meta' in path_lower and 'analysis' in path_lower:
            doc_type = 'meta_analysis'
        elif 'systematic' in path_lower:
            doc_type = 'systematic_review'
        else:
            doc_type = 'research_paper'
        
        metadata['document_type'] = doc_type
        
        # Extract specialty tags from path
        specialty_tags = []
        for specialty, keywords in self.medical_config.SPECIALTY_AREAS.items():
            if any(keyword in path_lower for keyword in keywords):
                specialty_tags.append(specialty)
        
        metadata['specialty_tags'] = specialty_tags
        
        # Extract category from directory structure
        parts = file_path.parts
        if len(parts) > 2:  # Has subdirectories
            # Use the immediate parent directory as category
            metadata['category'] = parts[-2].replace('_', ' ').title()
        
        return metadata
    
    def _finalize_metadata(self, metadata: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Finalize and clean metadata."""
        # Set defaults for required fields
        metadata.setdefault('title', file_path.stem.replace('_', ' ').title())
        metadata.setdefault('authors', [])
        metadata.setdefault('journal', 'Unknown')
        metadata.setdefault('publication_date', 'Unknown')
        metadata.setdefault('abstract', '')
        metadata.setdefault('keywords', [])
        metadata.setdefault('document_type', 'research_paper')
        metadata.setdefault('specialty_tags', [])
        metadata.setdefault('language', 'en')
        metadata.setdefault('category', 'general')
        
        # Clean up authors list
        if isinstance(metadata['authors'], str):
            metadata['authors'] = [metadata['authors']]
        metadata['authors'] = [a.strip() for a in metadata['authors'] if a.strip()]
        
        # Clean up specialty tags
        if isinstance(metadata['specialty_tags'], str):
            metadata['specialty_tags'] = [metadata['specialty_tags']]
        metadata['specialty_tags'] = [t.strip() for t in metadata['specialty_tags'] if t.strip()]
        
        # Clean up keywords
        if isinstance(metadata['keywords'], str):
            metadata['keywords'] = [k.strip() for k in metadata['keywords'].split(',')]
        metadata['keywords'] = [k.strip() for k in metadata['keywords'] if k.strip()]
        
        return metadata
    
    def _create_fallback_metadata(self, file_path: Path, provided_metadata: Optional[Dict]) -> Dict[str, Any]:
        """Create fallback metadata when extraction fails."""
        metadata = provided_metadata.copy() if provided_metadata else {}
        
        metadata.setdefault('title', file_path.stem.replace('_', ' ').title())
        metadata.setdefault('authors', [])
        metadata.setdefault('journal', 'Unknown')
        metadata.setdefault('publication_date', 'Unknown')
        metadata.setdefault('abstract', '')
        metadata.setdefault('keywords', [])
        metadata.setdefault('document_type', 'research_paper')
        metadata.setdefault('specialty_tags', ['transplant', 'infection'])  # Default for our domain
        metadata.setdefault('language', 'en')
        metadata.setdefault('category', 'general')
        
        return metadata

class DocumentQualityAssessor:
    """Assess document quality for medical relevance and completeness."""
    
    def __init__(self):
        """Initialize quality assessor."""
        self.medical_config = MedicalConfig()
    
    def assess(self, processed_text: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """
        Assess overall document quality.
        
        Args:
            processed_text: Processed text information
            metadata: Document metadata
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            criteria = self.medical_config.QUALITY_CRITERIA
            score = 0.0
            
            # Content length assessment
            score += self._assess_content_length(processed_text) * criteria['content_length_weight']
            
            # Section structure assessment
            score += self._assess_section_structure(processed_text) * criteria['section_structure_weight']
            
            # Metadata completeness assessment
            score += self._assess_metadata_completeness(metadata) * criteria['metadata_completeness_weight']
            
            # Medical relevance assessment
            score += self._assess_medical_relevance(processed_text, metadata) * criteria['medical_relevance_weight']
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default score
    
    def _assess_content_length(self, processed_text: Dict[str, Any]) -> float:
        """Assess content length appropriateness."""
        word_count = processed_text.get('word_count', 0)
        
        if word_count > 2000:
            return 1.0
        elif word_count > 1000:
            return 0.8
        elif word_count > 500:
            return 0.6
        elif word_count > 200:
            return 0.4
        else:
            return 0.2
    
    def _assess_section_structure(self, processed_text: Dict[str, Any]) -> float:
        """Assess document structure quality."""
        sections = processed_text.get('sections', {})
        
        # Expected sections for medical papers
        key_sections = ['abstract', 'introduction', 'methods', 'results', 'discussion']
        found_sections = sum(1 for section in key_sections if section in sections)
        
        if found_sections >= 4:
            return 1.0
        elif found_sections >= 3:
            return 0.8
        elif found_sections >= 2:
            return 0.6
        elif found_sections >= 1:
            return 0.4
        else:
            return 0.2
    
    def _assess_metadata_completeness(self, metadata: Dict[str, Any]) -> float:
        """Assess metadata completeness."""
        required_fields = ['title', 'authors', 'journal', 'publication_date']
        optional_fields = ['abstract', 'keywords', 'doi']
        
        required_score = sum(
            1 for field in required_fields 
            if metadata.get(field) and metadata[field] not in ['Unknown', '']
        ) / len(required_fields)
        
        optional_score = sum(
            1 for field in optional_fields 
            if metadata.get(field) and metadata[field] not in ['Unknown', '']
        ) / len(optional_fields)
        
        return 0.7 * required_score + 0.3 * optional_score
    
    def _assess_medical_relevance(self, processed_text: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Assess medical relevance for transplant infections domain."""
        score = 0.0
        
        # Check specialty tags
        specialty_tags = metadata.get('specialty_tags', [])
        relevant_specialties = ['transplant', 'infection', 'immunology']
        specialty_score = sum(1 for spec in relevant_specialties if spec in specialty_tags) / len(relevant_specialties)
        score += 0.4 * specialty_score
        
        # Check medical entity density in chunks
        chunks = processed_text.get('chunks', [])
        if chunks:
            entity_scores = []
            for chunk in chunks:
                entities = chunk.get('medical_entities', {})
                total_entities = sum(len(entity_list) for entity_list in entities.values())
                word_count = chunk.get('word_count', 1)
                entity_density = min(total_entities / word_count * 100, 1.0)
                entity_scores.append(entity_density)
            
            avg_entity_density = sum(entity_scores) / len(entity_scores) if entity_scores else 0
            score += 0.4 * avg_entity_density
        
        # Check for medical keywords in title and abstract
        text_to_check = (metadata.get('title', '') + ' ' + metadata.get('abstract', '')).lower()
        medical_keywords = [
            'transplant', 'infection', 'immunosuppression', 'graft', 'rejection',
            'bacterial', 'viral', 'fungal', 'antimicrobial', 'prophylaxis'
        ]
        
        keyword_score = sum(1 for keyword in medical_keywords if keyword in text_to_check) / len(medical_keywords)
        score += 0.2 * keyword_score
        
        return min(score, 1.0)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_single_document(file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to process a single document.
    
    Args:
        file_path: Path to document
        metadata: Optional metadata
        
    Returns:
        Processed document information
    """
    processor = DocumentTextProcessor()
    return processor.process_document(file_path)

def extract_document_metadata(file_path: str, provided_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to extract document metadata.
    
    Args:
        file_path: Path to document
        provided_metadata: Optional provided metadata
        
    Returns:
        Extracted metadata
    """
    extractor = MetadataExtractor()
    return extractor.extract(file_path, provided_metadata)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Demo functionality when run as main module."""
    import sys
    
    print("Document Processing Engine")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Process file provided as argument
        file_path = sys.argv[1]
        print(f"🔄 Processing: {file_path}")
        
        try:
            result = process_single_document(file_path)
            
            print(f"Processing complete!")
            print(f"   Sections: {result['section_count']}")
            print(f"   Chunks: {result['chunk_count']}")
            print(f"   Words: {result['word_count']}")
            
            if result['chunks']:
                print(f"   Sample chunk: {result['chunks'][0]['text'][:100]}...")
            
        except Exception as e:
            print(f" Processing failed: {e}")
    else:
        print("Usage: python document_processor.py <file_path>")
        print("   or import and use programmatically")