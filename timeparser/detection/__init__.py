"""
Temporal Detection Module

This module provides span detection for temporal expressions using:
1. tei2go - spaCy NER-based detection (optional, requires spaCy)
2. temporal_signals - Discourse marker and signal detection (regex-based)

Usage:
    from timeparser.detection import extract_temporal_spans, extract_temporal_signals
    
    # Extract TIMEX spans (requires spaCy + tei2go model)
    spans = extract_temporal_spans("The meeting is on July 20, 2023.")
    
    # Extract temporal signals/discourse markers
    signals = extract_temporal_signals("Later, the team reconvened.")
"""

# tei2go exports (spaCy-based NER detection)
from .tei2go import (
    extract_temporal_spans,
    extract_temporal_spans_batch,
    extract_temporal_spans_auto,
    extract_temporal_spans_fallback,
    load_tei2go_model,
    is_tei2go_available,
    get_available_languages,
    TemporalSpan,
    SPACY_AVAILABLE,
)

# temporal_signals exports (regex-based signal detection)
from .temporal_signals import (
    extract_temporal_signals,
    extract_temporal_signals_batch,
    TemporalSignal,
    TemporalRelation,
    SignalCategory,
    filter_by_relation,
    filter_by_category,
    filter_anaphoric,
    filter_discourse_markers,
    get_signal_lexicon,
    get_relation_for_signal,
    count_signals_by_relation,
    count_signals_by_category,
)

__all__ = [
    # tei2go
    "extract_temporal_spans",
    "extract_temporal_spans_batch",
    "extract_temporal_spans_auto",
    "extract_temporal_spans_fallback",
    "load_tei2go_model",
    "is_tei2go_available",
    "get_available_languages",
    "TemporalSpan",
    "SPACY_AVAILABLE",
    # temporal_signals
    "extract_temporal_signals",
    "extract_temporal_signals_batch",
    "TemporalSignal",
    "TemporalRelation",
    "SignalCategory",
    "filter_by_relation",
    "filter_by_category",
    "filter_anaphoric",
    "filter_discourse_markers",
    "get_signal_lexicon",
    "get_relation_for_signal",
    "count_signals_by_relation",
    "count_signals_by_category",
]


