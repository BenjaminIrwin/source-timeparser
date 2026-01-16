"""
TEI2GO Span Detection

This module provides a wrapper around the TEI2GO spaCy models for detecting
temporal expression spans that are later converted to SCATEX code.

spaCy and TEI2GO models are OPTIONAL. If not installed, the module gracefully
degrades to regex-based fallback detection.

TEI2GO models are available for multiple languages:
- en_tei2go: English
- fr_tei2go: French  
- de_tei2go: German
- es_tei2go: Spanish
- pt_tei2go: Portuguese

Installation:
    pip install spacy
    pip install https://huggingface.co/lxyuan/en_tei2go/resolve/main/en_tei2go-any-py3-none-any.whl
    
Or via spacy:
    python -m spacy download en_tei2go
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Try to import spacy (optional dependency)
try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Doc = None
    logger.info("spaCy not available - tei2go detection disabled, using regex fallback")

# Cache for loaded models
_model_cache: Dict[str, Any] = {}  # Any because spacy.Language may not be available


@dataclass
class TemporalSpan:
    """A detected temporal expression span."""
    text: str           # The temporal expression text
    start: int          # Start character offset in original text
    end: int            # End character offset in original text
    label: str          # Entity label (usually "TIMEX" or similar)
    sentence: str       # The containing sentence for context
    sentence_start: int # Start of sentence in original text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "sentence": self.sentence,
            "sentence_start": self.sentence_start
        }


def _get_model_name(lang: str) -> str:
    """Get the TEI2GO model name for a language."""
    model_map = {
        "en": "en_tei2go",
        "english": "en_tei2go",
        "fr": "fr_tei2go",
        "french": "fr_tei2go",
        "de": "de_tei2go",
        "german": "de_tei2go",
        "es": "es_tei2go",
        "spanish": "es_tei2go",
        "pt": "pt_tei2go",
        "portuguese": "pt_tei2go",
    }
    return model_map.get(lang.lower(), f"{lang}_tei2go")


def load_tei2go_model(lang: str = "en") -> Optional[Any]:
    """
    Load the TEI2GO spaCy model for a language.
    
    Args:
        lang: Language code (e.g., "en", "fr", "de")
        
    Returns:
        spaCy Language object, or None if model not available
    """
    if not SPACY_AVAILABLE:
        return None
    
    model_name = _get_model_name(lang)
    
    # Check cache first
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        nlp = spacy.load(model_name)
        
        # Add sentencizer if not present (TEI2GO models may not include it)
        if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
            logger.info(f"Added sentencizer to {model_name} pipeline")
        
        _model_cache[model_name] = nlp
        logger.info(f"Loaded TEI2GO model: {model_name}")
        return nlp
    except OSError as e:
        logger.warning(
            f"TEI2GO model '{model_name}' not found. "
            f"Install it with: pip install https://huggingface.co/lxyuan/{model_name}/resolve/main/{model_name}-any-py3-none-any.whl"
        )
        return None
    except Exception as e:
        logger.error(f"Error loading TEI2GO model '{model_name}': {e}")
        return None


def _get_sentence_for_span(doc: Any, span_start: int, span_end: int) -> Tuple[str, int]:
    """
    Get the sentence containing a character span.
    
    Returns:
        Tuple of (sentence_text, sentence_start_char)
    """
    for sent in doc.sents:
        if sent.start_char <= span_start and sent.end_char >= span_end:
            return (sent.text, sent.start_char)
    
    # Fallback: return context around the span
    context_chars = 100
    start = max(0, span_start - context_chars)
    end = min(len(doc.text), span_end + context_chars)
    return (doc.text[start:end], start)


def extract_temporal_spans(
    text: str, 
    lang: str = "en",
    nlp: Optional[Any] = None
) -> List[TemporalSpan]:
    """
    Extract temporal expression spans from text using TEI2GO.
    
    Args:
        text: The input text to analyze
        lang: Language code (e.g., "en", "fr", "de")
        nlp: Optional pre-loaded spaCy model (for efficiency in batch processing)
        
    Returns:
        List of TemporalSpan objects with detected temporal expressions
    """
    if not text or not text.strip():
        return []
    
    if not SPACY_AVAILABLE:
        logger.debug("spaCy not available, returning empty spans")
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_tei2go_model(lang)
    
    if nlp is None:
        logger.warning("No TEI2GO model available, returning empty spans")
        return []
    
    try:
        doc = nlp(text)
    except Exception as e:
        logger.error(f"Error processing text with TEI2GO: {e}")
        return []
    
    spans = []
    
    for ent in doc.ents:
        # TEI2GO typically labels temporal expressions as TIMEX, DATE, TIME, DURATION, SET
        # The exact labels depend on the model version
        if ent.label_ in ("TIMEX", "DATE", "TIME", "DURATION", "SET", "TIMEX3"):
            sentence, sent_start = _get_sentence_for_span(doc, ent.start_char, ent.end_char)
            
            spans.append(TemporalSpan(
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                label=ent.label_,
                sentence=sentence,
                sentence_start=sent_start
            ))
    
    return spans


def extract_temporal_spans_batch(
    texts: List[str],
    lang: str = "en",
    batch_size: int = 100
) -> List[List[TemporalSpan]]:
    """
    Extract temporal expressions from multiple texts efficiently using batching.
    
    Args:
        texts: List of input texts
        lang: Language code
        batch_size: Number of texts to process at once
        
    Returns:
        List of lists of TemporalSpan objects (one list per input text)
    """
    if not SPACY_AVAILABLE:
        logger.debug("spaCy not available, returning empty spans for all texts")
        return [[] for _ in texts]
    
    nlp = load_tei2go_model(lang)
    
    if nlp is None:
        logger.warning("No TEI2GO model available, returning empty spans")
        return [[] for _ in texts]
    
    results = []
    
    # Process in batches using spaCy's pipe
    for doc in nlp.pipe(texts, batch_size=batch_size):
        spans = []
        for ent in doc.ents:
            if ent.label_ in ("TIMEX", "DATE", "TIME", "DURATION", "SET", "TIMEX3"):
                sentence, sent_start = _get_sentence_for_span(doc, ent.start_char, ent.end_char)
                
                spans.append(TemporalSpan(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                    sentence=sentence,
                    sentence_start=sent_start
                ))
        results.append(spans)
    
    return results


# =============================================================================
# Fallback: Regex-based temporal detection (when no model available)
# =============================================================================

# Common temporal patterns
TEMPORAL_PATTERNS = [
    # ISO dates
    (r'\d{4}-\d{2}-\d{2}', 'DATE'),
    (r'\d{4}/\d{2}/\d{2}', 'DATE'),
    (r'\d{2}/\d{2}/\d{4}', 'DATE'),
    (r'\d{2}-\d{2}-\d{4}', 'DATE'),
    
    # Written dates
    (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', 'DATE'),
    (r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b', 'DATE'),
    (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', 'DATE'),
    
    # Month and year
    (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', 'DATE'),
    
    # Relative dates
    (r'\b(?:today|tomorrow|yesterday)\b', 'DATE'),
    (r'\b(?:this|last|next)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'DATE'),
    (r'\b\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|from\s+now|later|earlier)\b', 'DURATION'),
    
    # Times
    (r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b', 'TIME'),
    (r'\b(?:noon|midnight|morning|afternoon|evening|night)\b', 'TIME'),
    
    # Decades/centuries
    (r'\b(?:the\s+)?\d{4}s\b', 'DATE'),
    (r'\b(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+century\b', 'DATE'),
    
    # Quarters
    (r'\bQ[1-4]\s+\d{4}\b', 'DATE'),
    
    # Duration expressions
    (r'\bfor\s+\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b', 'DURATION'),
    (r'\b\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+(?:long|old)\b', 'DURATION'),
]


def extract_temporal_spans_fallback(text: str) -> List[TemporalSpan]:
    """
    Fallback regex-based temporal extraction when TEI2GO is not available.
    
    This is less accurate than TEI2GO but provides basic functionality.
    
    Args:
        text: Input text
        
    Returns:
        List of TemporalSpan objects
    """
    if not text:
        return []
    
    spans = []
    seen_ranges = set()  # Avoid overlapping matches
    
    for pattern, label in TEMPORAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()
            
            # Check for overlap with existing spans
            overlaps = False
            for seen_start, seen_end in seen_ranges:
                if start < seen_end and end > seen_start:
                    overlaps = True
                    break
            
            if not overlaps:
                # Get context (simple sentence approximation)
                context_start = max(0, text.rfind('.', 0, start) + 1)
                context_end = text.find('.', end)
                if context_end == -1:
                    context_end = len(text)
                else:
                    context_end += 1
                
                sentence = text[context_start:context_end].strip()
                
                spans.append(TemporalSpan(
                    text=match.group(),
                    start=start,
                    end=end,
                    label=label,
                    sentence=sentence,
                    sentence_start=context_start
                ))
                seen_ranges.add((start, end))
    
    # Sort by position
    spans.sort(key=lambda s: s.start)
    
    return spans


def extract_temporal_spans_auto(
    text: str,
    lang: str = "en",
    use_fallback: bool = True
) -> List[TemporalSpan]:
    """
    Extract temporal spans, automatically falling back to regex if TEI2GO unavailable.
    
    Args:
        text: Input text
        lang: Language code
        use_fallback: Whether to use regex fallback if model not available
        
    Returns:
        List of TemporalSpan objects
    """
    # Try TEI2GO first
    spans = extract_temporal_spans(text, lang)
    
    # If no spans and TEI2GO might not be loaded, try fallback
    if not spans and use_fallback:
        nlp = load_tei2go_model(lang)
        if nlp is None:
            logger.info("Using regex fallback for temporal extraction")
            spans = extract_temporal_spans_fallback(text)
    
    return spans


# =============================================================================
# Utility Functions
# =============================================================================

def is_tei2go_available(lang: str = "en") -> bool:
    """Check if TEI2GO model is available for a language."""
    if not SPACY_AVAILABLE:
        return False
    nlp = load_tei2go_model(lang)
    return nlp is not None


def get_available_languages() -> List[str]:
    """Get list of languages with available TEI2GO models."""
    if not SPACY_AVAILABLE:
        return []
    available = []
    for lang in ["en", "fr", "de", "es", "pt"]:
        if is_tei2go_available(lang):
            available.append(lang)
    return available


