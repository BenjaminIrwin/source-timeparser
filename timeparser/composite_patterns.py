"""
Composite Temporal Pattern Recognition

This module handles multi-part temporal expressions that combine:
- Range indicators (between, from...to) with dates
- Modifiers (early, late, mid) with time intervals
- Approximation markers (around, circa) with temporal expressions

These patterns are checked BEFORE individual date parsing to ensure
the entire expression is captured as a single SCATEX unit.

Pattern Priority (highest first):
1. Modified time-of-day with date: "early in the morning of October 7, 2023"
2. Range patterns: "between October 7 and July 1", "from X to Y"
3. Approximate: "around 6 PM", "circa 1990"
4. Modified time-of-day: "early morning", "late evening"
5. Modified date unit: "early October", "late 2023"
6. Bounded: "since X", "until X", "before X", "after X"
7. Dash range: "X - Y" (lowest priority, most ambiguous)
"""

import logging
import regex as re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Any

from .scatex import (
    TemporalExpression,
    Between, Before, After, Intersection,
    TimeOfDay, TimeOfDayType,
    Day, Month, Year, Hour, Minute,
    ModifiedInterval, Approximate,
    Period, Unit,
    Unknown,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

TIME_OF_DAY_MAP = {
    'morning': TimeOfDayType.MORNING,
    'afternoon': TimeOfDayType.AFTERNOON,
    'evening': TimeOfDayType.EVENING,
    'night': TimeOfDayType.NIGHT,
    'dawn': TimeOfDayType.DAWN,
    'noon': TimeOfDayType.NOON,
    'midnight': TimeOfDayType.MIDNIGHT,
}

MONTH_NAME_MAP = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}

POSITION_MODIFIERS = {'early', 'late', 'mid'}


# =============================================================================
# Pattern Definition
# =============================================================================

@dataclass
class CompositePattern:
    """Definition of a composite temporal pattern."""
    name: str
    regex: re.Pattern
    handler: str  # Name of handler method
    priority: int


# =============================================================================
# CompositePatternParser
# =============================================================================

class CompositePatternParser:
    """
    Parser for composite temporal expressions.
    
    This parser attempts to match composite patterns before falling back
    to standard date parsing, ensuring expressions like "between X and Y"
    are captured as unified SCATEX expressions.
    """
    
    def __init__(self):
        self._patterns = self._compile_patterns()
        # Sort by priority (highest first)
        self._patterns.sort(key=lambda p: p.priority, reverse=True)
        self._parsing_depth = 0
        self._max_depth = 3
    
    def _compile_patterns(self) -> List[CompositePattern]:
        """Compile all pattern definitions."""
        patterns = []
        
        # =====================================================================
        # MODIFIED TIME-OF-DAY WITH DATE (Priority: 110)
        # "early in the morning of October 7, 2023"
        # "late in the evening of October 7"
        # =====================================================================
        patterns.append(CompositePattern(
            name='modifier_time_of_day_date',
            regex=re.compile(
                r'^(early|late|mid)\s+(?:in\s+the\s+)?(morning|afternoon|evening|night|dawn|noon)\s+(?:of\s+)?(.+)$',
                re.IGNORECASE
            ),
            handler='handle_modified_time_of_day_with_date',
            priority=110,
        ))
        
        # "early morning of October 7, 2023" (more compact form)
        patterns.append(CompositePattern(
            name='modifier_time_of_day_date_compact',
            regex=re.compile(
                r'^(early|late|mid)[- ]?(morning|afternoon|evening|night|dawn|noon)\s+(?:of\s+)?(.+)$',
                re.IGNORECASE
            ),
            handler='handle_modified_time_of_day_with_date',
            priority=105,
        ))
        
        # =====================================================================
        # RANGE PATTERNS (Priority: 100)
        # "between October 7 and July 1"
        # "from October 7 to July 1"
        # =====================================================================
        patterns.append(CompositePattern(
            name='between_and',
            regex=re.compile(
                r'^between\s+(.+?)\s+and\s+(.+?)$',
                re.IGNORECASE
            ),
            handler='handle_between_range',
            priority=100,
        ))
        
        patterns.append(CompositePattern(
            name='from_to',
            regex=re.compile(
                r'^from\s+(.+?)\s+(?:to|until|through|till)\s+(.+?)$',
                re.IGNORECASE
            ),
            handler='handle_between_range',
            priority=100,
        ))
        
        # "Jan 1st through Feb 15th" or "January 1 to March 15"
        # Standalone "X through/to Y" without "from"
        patterns.append(CompositePattern(
            name='through_range',
            regex=re.compile(
                r'^(.+?)\s+(?:through|thru|to)\s+(.+?)$',
                re.IGNORECASE
            ),
            handler='handle_through_range',
            priority=98,
        ))
        
        # =====================================================================
        # APPROXIMATE PATTERNS (Priority: 95)
        # "around 6 PM", "circa 1990", "approximately 3 weeks ago"
        # =====================================================================
        patterns.append(CompositePattern(
            name='approximate',
            regex=re.compile(
                r'^(?:around|approximately|about|roughly|circa|c\.)\s+(.+)$',
                re.IGNORECASE
            ),
            handler='handle_approximate',
            priority=95,
        ))
        
        # =====================================================================
        # MODIFIED TIME-OF-DAY (standalone) (Priority: 90)
        # "early morning", "late evening", "mid-afternoon"
        # =====================================================================
        patterns.append(CompositePattern(
            name='modifier_time_of_day',
            regex=re.compile(
                r'^(early|late|mid)[- ]?(morning|afternoon|evening|night|dawn|noon|midnight)$',
                re.IGNORECASE
            ),
            handler='handle_modified_time_of_day',
            priority=90,
        ))
        
        # =====================================================================
        # MODIFIED DATE UNIT (Priority: 85)
        # "early October", "late 2023", "mid-January"
        # =====================================================================
        patterns.append(CompositePattern(
            name='modifier_month',
            regex=re.compile(
                r'^(early|late|mid)[- ]?(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?$',
                re.IGNORECASE
            ),
            handler='handle_modified_month',
            priority=85,
        ))
        
        patterns.append(CompositePattern(
            name='modifier_year',
            regex=re.compile(
                r'^(early|late|mid)[- ]?(\d{4})$',
                re.IGNORECASE
            ),
            handler='handle_modified_year',
            priority=85,
        ))
        
        # =====================================================================
        # BOUNDED PATTERNS (Priority: 80)
        # "since October 7, 2023", "until July 1", "before 2020", "after 2020"
        # =====================================================================
        patterns.append(CompositePattern(
            name='since_date',
            regex=re.compile(
                r'^since\s+(.+)$',
                re.IGNORECASE
            ),
            handler='handle_since',
            priority=80,
        ))
        
        patterns.append(CompositePattern(
            name='until_date',
            regex=re.compile(
                r'^(?:until|up\s+to|through|till)\s+(.+)$',
                re.IGNORECASE
            ),
            handler='handle_until',
            priority=80,
        ))
        
        patterns.append(CompositePattern(
            name='before_date',
            regex=re.compile(
                r'^before\s+(.+)$',
                re.IGNORECASE
            ),
            handler='handle_before',
            priority=80,
        ))
        
        patterns.append(CompositePattern(
            name='after_date',
            regex=re.compile(
                r'^after\s+(.+)$',
                re.IGNORECASE
            ),
            handler='handle_after',
            priority=80,
        ))
        
        # =====================================================================
        # DASH RANGE (Priority: 50) - lowest priority, most ambiguous
        # "2020 - 2023", "October 7 - July 1"
        # =====================================================================
        patterns.append(CompositePattern(
            name='dash_range',
            regex=re.compile(
                r'^(.+?)\s*[-–—]\s*(.+?)$',  # Supports hyphen, en-dash, em-dash
                re.IGNORECASE
            ),
            handler='handle_dash_range',
            priority=50,
        ))
        
        return patterns
    
    def try_parse(self, date_string: str, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """
        Attempt to parse a composite pattern.
        
        Args:
            date_string: The date string to parse
            settings: Parser settings object
            
        Returns:
            Tuple of (expression, period) if matched, None otherwise.
        """
        if not date_string or not date_string.strip():
            return None
        
        date_string = date_string.strip()
        
        for pattern in self._patterns:
            match = pattern.regex.match(date_string)
            if match:
                handler = getattr(self, pattern.handler, None)
                if handler:
                    try:
                        result = handler(match, settings)
                        if result:
                            logger.debug(f"Composite pattern '{pattern.name}' matched: {date_string}")
                            return result
                    except Exception as e:
                        logger.debug(f"Handler {pattern.handler} failed for '{date_string}': {e}")
                        continue
        
        return None
    
    def _parse_subexpression(self, text: str, settings: Any, depth: int = 0) -> Optional[TemporalExpression]:
        """
        Recursively parse a sub-expression (e.g., the dates in "between X and Y").
        
        Uses the main timeparser.parse() function but with depth limiting to prevent
        infinite recursion.
        
        Args:
            text: The text to parse
            settings: Parser settings
            depth: Current recursion depth
            
        Returns:
            Parsed SCATEX expression or None
        """
        if depth > self._max_depth:
            logger.debug(f"Max recursion depth exceeded parsing: {text}")
            return None
        
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Import here to avoid circular import
        import timeparser
        
        # Temporarily track depth
        old_depth = self._parsing_depth
        self._parsing_depth = depth + 1
        
        try:
            # Try to parse with timeparser
            # We need to pass proper settings - extract the dict if it has _settings
            if hasattr(settings, '_settings'):
                settings_dict = settings._settings
            else:
                settings_dict = settings if isinstance(settings, dict) else None
            
            result = timeparser.parse(text, languages=['en'], settings=settings_dict)
            
            if result and not isinstance(result, Unknown):
                return result
            
            return None
            
        except Exception as e:
            logger.debug(f"Sub-expression parsing failed for '{text}': {e}")
            return None
        finally:
            self._parsing_depth = old_depth
    
    # =========================================================================
    # Handler Methods
    # =========================================================================
    
    def handle_between_range(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'between X and Y' and 'from X to Y' patterns."""
        start_text = match.group(1).strip()
        end_text = match.group(2).strip()
        
        start_expr = self._parse_subexpression(start_text, settings)
        end_expr = self._parse_subexpression(end_text, settings)
        
        if start_expr and end_expr:
            return (Between(start_interval=start_expr, end_interval=end_expr), "day")
        
        return None
    
    def handle_through_range(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """
        Handle 'X through Y' and 'X to Y' patterns (without 'from').
        
        Examples:
            - "Jan 1st through Feb 15th"
            - "January to March"
            - "2020 thru 2023"
        
        This is more ambiguous than 'from X to Y', so we require both parts
        to parse as valid temporal expressions of compatible types.
        """
        start_text = match.group(1).strip()
        end_text = match.group(2).strip()
        
        # Quickly reject obvious non-temporal phrases
        # "to" is very common in non-temporal contexts
        non_temporal_starts = {'going', 'went', 'want', 'wanted', 'need', 'needed', 'have', 'had', 'get', 'got'}
        first_word = start_text.split()[0].lower() if start_text else ''
        if first_word in non_temporal_starts:
            return None
        
        start_expr = self._parse_subexpression(start_text, settings)
        end_expr = self._parse_subexpression(end_text, settings)
        
        if not start_expr or not end_expr:
            return None
        
        # Validate types are compatible for a range
        # Both should be similar temporal granularity
        compatible_types = {
            Year: {Year},
            Month: {Month},
            Day: {Day},
            Hour: {Hour},
            Minute: {Minute},
        }
        
        start_type = type(start_expr)
        end_type = type(end_expr)
        
        # Check if types are compatible
        if start_type in compatible_types:
            if end_type not in compatible_types.get(start_type, set()):
                return None
        elif start_type != end_type:
            # For other types, require exact match
            return None
        
        return (Between(start_interval=start_expr, end_interval=end_expr), "day")
    
    def handle_modified_time_of_day_with_date(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'early in the morning of October 7, 2023'."""
        modifier = match.group(1).lower()  # early, late, mid
        time_of_day = match.group(2).lower()  # morning, afternoon, etc.
        date_text = match.group(3).strip()
        
        if time_of_day not in TIME_OF_DAY_MAP:
            return None
        
        # Parse the date part
        date_expr = self._parse_subexpression(date_text, settings)
        if not date_expr:
            return None
        
        # Build: ModifiedInterval(Intersection([date, TimeOfDay]), modifier)
        intersection = Intersection(intervals=[
            date_expr,
            TimeOfDay(type=TIME_OF_DAY_MAP[time_of_day])
        ])
        
        modified = ModifiedInterval(interval=intersection, position=modifier)
        return (modified, "time")
    
    def handle_modified_time_of_day(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle standalone 'early morning', 'late evening', etc."""
        modifier = match.group(1).lower()
        time_of_day = match.group(2).lower()
        
        if time_of_day not in TIME_OF_DAY_MAP:
            return None
        
        tod_expr = TimeOfDay(type=TIME_OF_DAY_MAP[time_of_day])
        modified = ModifiedInterval(interval=tod_expr, position=modifier)
        return (modified, "time")
    
    def handle_modified_month(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'early October', 'late January 2023', etc."""
        modifier = match.group(1).lower()
        month_name = match.group(2).lower()
        year_str = match.group(3) if match.lastindex >= 3 else None
        
        if month_name not in MONTH_NAME_MAP:
            return None
        
        month_num = MONTH_NAME_MAP[month_name]
        year = int(year_str) if year_str else None
        
        month_expr = Month(month=month_num, year=year)
        modified = ModifiedInterval(interval=month_expr, position=modifier)
        return (modified, "month")
    
    def handle_modified_year(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'early 2023', 'late 2020', 'mid-2021'."""
        modifier = match.group(1).lower()
        year = int(match.group(2))
        
        year_expr = Year(digits=year)
        modified = ModifiedInterval(interval=year_expr, position=modifier)
        return (modified, "year")
    
    def handle_approximate(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'around X', 'approximately X', 'circa X'."""
        inner_text = match.group(1).strip()
        inner_expr = self._parse_subexpression(inner_text, settings)
        
        if inner_expr:
            # Determine appropriate margin based on the type of inner expression
            margin = self._get_default_margin(inner_expr)
            return (Approximate(interval=inner_expr, margin=margin), "day")
        
        return None
    
    def _get_default_margin(self, expr: TemporalExpression) -> Optional[Period]:
        """Get appropriate margin for an approximate expression based on its granularity."""
        if isinstance(expr, Year):
            return Period(unit=Unit.YEAR, value=5)  # circa 1990 = 1985-1995
        elif isinstance(expr, Month):
            return Period(unit=Unit.WEEK, value=2)  # around October = +/- 2 weeks
        elif isinstance(expr, Day):
            return Period(unit=Unit.DAY, value=2)  # around Oct 7 = +/- 2 days
        elif isinstance(expr, Hour):
            return Period(unit=Unit.HOUR, value=1)  # around 6 PM = +/- 1 hour
        elif isinstance(expr, Minute):
            return Period(unit=Unit.MINUTE, value=15)  # around 6:30 = +/- 15 min
        else:
            return None  # Use default 10% margin
    
    def handle_since(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'since X' pattern."""
        date_text = match.group(1).strip()
        date_expr = self._parse_subexpression(date_text, settings)
        
        if date_expr:
            return (After(interval=date_expr), "day")
        
        return None
    
    def handle_until(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'until X', 'up to X', 'through X' patterns."""
        date_text = match.group(1).strip()
        date_expr = self._parse_subexpression(date_text, settings)
        
        if date_expr:
            return (Before(interval=date_expr), "day")
        
        return None
    
    def handle_before(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'before X' pattern."""
        date_text = match.group(1).strip()
        date_expr = self._parse_subexpression(date_text, settings)
        
        if date_expr:
            return (Before(interval=date_expr), "day")
        
        return None
    
    def handle_after(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """Handle 'after X' pattern."""
        date_text = match.group(1).strip()
        date_expr = self._parse_subexpression(date_text, settings)
        
        if date_expr:
            return (After(interval=date_expr), "day")
        
        return None
    
    def handle_dash_range(self, match: re.Match, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
        """
        Handle 'X - Y' dash range patterns.
        
        This is the lowest priority pattern because dashes are ambiguous.
        We add extra validation to ensure both parts look like temporal expressions.
        """
        start_text = match.group(1).strip()
        end_text = match.group(2).strip()
        
        # Extra validation: both parts should be parseable and make sense as a range
        start_expr = self._parse_subexpression(start_text, settings)
        end_expr = self._parse_subexpression(end_text, settings)
        
        if not start_expr or not end_expr:
            return None
        
        # Validate that the types are compatible for a range
        # (both should be same granularity: Year-Year, Month-Month, Day-Day, etc.)
        if type(start_expr) != type(end_expr):
            # Allow some compatible combinations
            compatible = {
                (Year, Year), (Month, Month), (Day, Day),
                (Hour, Hour), (Minute, Minute),
            }
            if (type(start_expr), type(end_expr)) not in compatible:
                return None
        
        return (Between(start_interval=start_expr, end_interval=end_expr), "day")


# =============================================================================
# Singleton Instance
# =============================================================================

composite_parser = CompositePatternParser()


# =============================================================================
# Span Extraction Patterns (DEPRECATED)
# =============================================================================
# 
# NOTE: These regex patterns are superseded by merge_signals_with_spans() which
# uses already-detected signals and spans to form composites. This approach is
# more robust because it leverages the signal lexicon rather than trying to
# anticipate every possible phrasing.
#
# The patterns below are kept for reference but should not be used.
# Use merge_signals_with_spans() instead.

COMPOSITE_SPAN_PATTERNS = []  # Disabled - use merge_signals_with_spans() instead

# Old patterns for reference:
# COMPOSITE_SPAN_PATTERNS_OLD = [
#     # "between X and Y" - most reliable
#     re.compile(r'\b(between\s+.+?\s+and\s+.+?)(?:\.|,|\s+(?:according|when|where|while|after|before|during|the\s+\w+\s+said)|\s*$)', re.IGNORECASE),
#     # "from X to/until/through Y"
#     re.compile(r'\b(from\s+.+?\s+(?:to|until|through|till)\s+.+?)(?:\.|,|\s+(?:according|when|where|while|the\s+\w+\s+said)|\s*$)', re.IGNORECASE),
#     # etc.
# ]


@dataclass
class CompositeSpan:
    """A detected composite temporal span in text."""
    text: str
    start: int
    end: int
    expression: TemporalExpression
    pattern_name: str


def extract_composite_spans(text: str, settings: Any = None) -> List[CompositeSpan]:
    """
    DEPRECATED: Use merge_signals_with_spans() instead.
    
    This regex-based approach is superseded by signal-guided span merging,
    which is more robust because it uses already-detected signals and spans.
    
    This function now returns an empty list. For composite detection, use:
    
        from timeparser import extract_temporal_spans, extract_temporal_signals
        from timeparser.composite_patterns import merge_signals_with_spans
        
        spans = extract_temporal_spans(text)
        signals = extract_temporal_signals(text)
        result = merge_signals_with_spans(text, spans, signals)
        composites = result.composites
    
    Args:
        text: The text to scan
        settings: Optional parser settings (uses defaults if None)
        
    Returns:
        Empty list (function is deprecated).
    """
    import warnings
    warnings.warn(
        "extract_composite_spans() is deprecated. Use merge_signals_with_spans() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return []  # Return empty - use merge_signals_with_spans() instead
    if not text or not text.strip():
        return []
    
    if settings is None:
        from timeparser.conf import Settings
        settings = Settings()
    
    spans = []
    seen_ranges: set = set()  # Track (start, end) to avoid duplicates
    
    # Try each pattern
    for pattern in COMPOSITE_SPAN_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(1).strip()
            start = match.start(1)
            end = match.end(1)
            
            # Skip if overlapping with existing span
            overlaps = False
            for seen_start, seen_end in seen_ranges:
                if start < seen_end and end > seen_start:
                    overlaps = True
                    break
            if overlaps:
                continue
            
            # Try to parse the matched text
            result = composite_parser.try_parse(matched_text, settings)
            if result:
                expr, period = result
                spans.append(CompositeSpan(
                    text=matched_text,
                    start=start,
                    end=end,
                    expression=expr,
                    pattern_name=pattern.pattern[:50] + "..."
                ))
                seen_ranges.add((start, end))
    
    # Sort by position
    spans.sort(key=lambda s: s.start)
    
    return spans


def filter_spans_covered_by_composites(
    spans: List[Any],
    composite_spans: List[CompositeSpan]
) -> List[Any]:
    """
    Filter out spans that are covered by composite spans.
    
    Use this to remove individual date spans (like "October 7", "July 1")
    that are part of a composite pattern (like "between October 7 and July 1").
    
    Args:
        spans: List of spans (e.g., from TEI2GO) with .start and .end attributes
        composite_spans: List of CompositeSpan objects
        
    Returns:
        Filtered list of spans not covered by any composite span.
    """
    if not composite_spans:
        return spans
    
    filtered = []
    for span in spans:
        # Check if this span is covered by any composite
        is_covered = False
        for comp in composite_spans:
            if span.start >= comp.start and span.end <= comp.end:
                is_covered = True
                break
        
        if not is_covered:
            filtered.append(span)
    
    return filtered


# =============================================================================
# Signal-Guided Span Merging
# =============================================================================

# Maximum character gap between signal end and span start to consider them adjacent
MAX_ADJACENCY_GAP = 30

# Signal relations that map to After() SCATEX type
BEGINNING_RELATIONS = {'beginning', 'after', 'immediately_after'}

# Signal relations that map to Before() SCATEX type  
BEFORE_RELATIONS = {'before', 'ending', 'immediately_before'}

# Signal text patterns that indicate "between X and Y" constructs
BETWEEN_SIGNAL_TEXTS = {'between'}

# Signal texts that can form "X to Y" range patterns (like "from X to Y")
RANGE_SIGNAL_TEXTS = {'from'}

# Connectors for range patterns
RANGE_CONNECTORS = {'to', 'until', 'through', 'till', 'thru'}


@dataclass
class MergeResult:
    """Result of signal-guided span merging."""
    composites: List[CompositeSpan]
    remaining_spans: List[Any]  # Spans not merged
    remaining_signals: List[Any]  # Signals not merged


def merge_signals_with_spans(
    text: str,
    spans: List[Any],
    signals: List[Any],
    settings: Any = None
) -> MergeResult:
    """
    Merge signals with adjacent spans to form composite expressions.
    
    This is a more robust approach than regex-based pattern matching because
    it uses already-detected signals and spans and combines them based on
    semantics and position.
    
    Rules:
    - 'beginning' relation (since, after, from) + adjacent span → After(span)
    - 'before' relation (before, prior to, until) + adjacent span → Before(span)
    - 'between' signal + span + 'and' + span → Between(span1, span2)
    - Adjacent spans with dash between them → Between(span1, span2)
    
    Args:
        text: The full text being analyzed
        spans: List of temporal spans (e.g., from TEI2GO) with .start, .end, .text
        signals: List of temporal signals with .start, .end, .text, .relation
        settings: Optional parser settings
        
    Returns:
        MergeResult with composites, remaining_spans, remaining_signals
    """
    if settings is None:
        from timeparser.conf import Settings
        settings = Settings()
    
    composites: List[CompositeSpan] = []
    used_span_indices: set = set()
    used_signal_indices: set = set()
    
    # Sort spans and signals by position
    sorted_spans = sorted(enumerate(spans), key=lambda x: x[1].start)
    sorted_signals = sorted(enumerate(signals), key=lambda x: x[1].start)
    
    # =========================================================================
    # Pass 1: Handle dash ranges FIRST (highest priority, unambiguous)
    # e.g., "2008-2009" → Between(Year(2008), Year(2009))
    # This must run before signal merging to prevent "from 2008-2009" from
    # being incorrectly parsed as After(2008) with leftover "2009"
    # =========================================================================
    dash_composites = _find_dash_ranges(text, sorted_spans, used_span_indices, settings)
    for composite, span_indices in dash_composites:
        composites.append(composite)
        used_span_indices.update(span_indices)
    
    # =========================================================================
    # Pass 2: Handle "between X and Y" patterns
    # =========================================================================
    for sig_idx, signal in sorted_signals:
        if sig_idx in used_signal_indices:
            continue
            
        signal_text_lower = signal.text.lower().strip()
        
        if signal_text_lower in BETWEEN_SIGNAL_TEXTS:
            # Look for two spans after this signal with "and" between them
            result = _try_merge_between(
                text, signal, sig_idx, sorted_spans, used_span_indices, settings
            )
            if result:
                composite, span_indices = result
                composites.append(composite)
                used_signal_indices.add(sig_idx)
                used_span_indices.update(span_indices)
    
    # =========================================================================
    # Pass 3: Handle "from X to Y" range patterns
    # Must run before signal+span merging to prevent "from X" → After(X)
    # =========================================================================
    for sig_idx, signal in sorted_signals:
        if sig_idx in used_signal_indices:
            continue
        
        signal_text_lower = signal.text.lower().strip()
        
        if signal_text_lower in RANGE_SIGNAL_TEXTS:
            # Try to find "from X to Y" pattern
            result = _try_merge_from_to_range(
                text, signal, sig_idx, sorted_spans, used_span_indices, settings
            )
            if result:
                composite, span_indices = result
                composites.append(composite)
                used_signal_indices.add(sig_idx)
                used_span_indices.update(span_indices)
    
    # =========================================================================
    # Pass 4: Handle signal + adjacent span patterns (since X, before X, etc.)
    # =========================================================================
    for sig_idx, signal in sorted_signals:
        if sig_idx in used_signal_indices:
            continue
        
        relation_value = signal.relation.value if hasattr(signal.relation, 'value') else str(signal.relation)
        
        # Check if this signal indicates a beginning or before relation
        if relation_value in BEGINNING_RELATIONS:
            result = _try_merge_signal_span(
                text, signal, sig_idx, sorted_spans, used_span_indices, 
                After, settings
            )
            if result:
                composite, span_idx = result
                composites.append(composite)
                used_signal_indices.add(sig_idx)
                used_span_indices.add(span_idx)
                
        elif relation_value in BEFORE_RELATIONS:
            result = _try_merge_signal_span(
                text, signal, sig_idx, sorted_spans, used_span_indices,
                Before, settings
            )
            if result:
                composite, span_idx = result
                composites.append(composite)
                used_signal_indices.add(sig_idx)
                used_span_indices.add(span_idx)
    
    # =========================================================================
    # Collect remaining (unmerged) spans and signals
    # =========================================================================
    remaining_spans = [span for idx, span in enumerate(spans) if idx not in used_span_indices]
    remaining_signals = [sig for idx, sig in enumerate(signals) if idx not in used_signal_indices]
    
    # Sort composites by position
    composites.sort(key=lambda c: c.start)
    
    return MergeResult(
        composites=composites,
        remaining_spans=remaining_spans,
        remaining_signals=remaining_signals
    )


def _try_merge_between(
    text: str,
    signal: Any,
    signal_idx: int,
    sorted_spans: List[Tuple[int, Any]],
    used_indices: set,
    settings: Any
) -> Optional[Tuple[CompositeSpan, List[int]]]:
    """
    Try to merge a 'between' signal with two spans connected by 'and'.
    
    Pattern: "between [span1] and [span2]"
    """
    # Find first span after signal
    first_span = None
    first_span_idx = None
    
    for idx, span in sorted_spans:
        if idx in used_indices:
            continue
        # Span must start after signal ends, within adjacency window
        gap = span.start - signal.end
        if 0 <= gap <= MAX_ADJACENCY_GAP:
            first_span = span
            first_span_idx = idx
            break
    
    if not first_span:
        return None
    
    # Look for "and" between first span and a second span
    search_start = first_span.end
    search_region = text[search_start:search_start + 50]  # Look ahead
    
    and_match = re.search(r'\s+and\s+', search_region, re.IGNORECASE)
    if not and_match:
        return None
    
    and_end_pos = search_start + and_match.end()
    
    # Find second span after "and"
    second_span = None
    second_span_idx = None
    
    for idx, span in sorted_spans:
        if idx in used_indices or idx == first_span_idx:
            continue
        # Span must start near where "and" ends
        gap = span.start - and_end_pos
        if -2 <= gap <= 10:  # Allow small flexibility
            second_span = span
            second_span_idx = idx
            break
    
    if not second_span:
        return None
    
    # Parse both spans
    first_expr = _parse_span_to_scatex(first_span.text, settings)
    second_expr = _parse_span_to_scatex(second_span.text, settings)
    
    if not first_expr or not second_expr:
        return None
    
    # Propagate context from first to second
    # e.g., "between October 6 and 7" → 7 inherits October from first
    second_expr = _propagate_context(first_expr, second_expr)
    
    # Also propagate year if one has it and the other doesn't
    first_expr, second_expr = _propagate_year_in_range(first_expr, second_expr)
    
    # Create Between expression
    between_expr = Between(start_interval=first_expr, end_interval=second_expr)
    
    # Full text from signal start to second span end
    full_text = text[signal.start:second_span.end]
    
    composite = CompositeSpan(
        text=full_text,
        start=signal.start,
        end=second_span.end,
        expression=between_expr,
        pattern_name="signal_merge:between_and"
    )
    
    return (composite, [first_span_idx, second_span_idx])


def _try_merge_from_to_range(
    text: str,
    signal: Any,
    signal_idx: int,
    sorted_spans: List[Tuple[int, Any]],
    used_indices: set,
    settings: Any
) -> Optional[Tuple[CompositeSpan, List[int]]]:
    """
    Try to merge a 'from' signal with two spans connected by 'to/until/through'.
    
    Pattern: "from [span1] to [span2]"
    """
    # Find first span after signal
    first_span = None
    first_span_idx = None
    
    for idx, span in sorted_spans:
        if idx in used_indices:
            continue
        gap = span.start - signal.end
        if 0 <= gap <= MAX_ADJACENCY_GAP:
            first_span = span
            first_span_idx = idx
            break
    
    if not first_span:
        return None
    
    # Look for "to/until/through" between first span and a second span
    search_start = first_span.end
    search_region = text[search_start:search_start + 50]
    
    # Build regex for range connectors
    connector_pattern = r'\s+(' + '|'.join(RANGE_CONNECTORS) + r')\s+'
    connector_match = re.search(connector_pattern, search_region, re.IGNORECASE)
    if not connector_match:
        return None
    
    connector_end_pos = search_start + connector_match.end()
    
    # Find second span after connector
    second_span = None
    second_span_idx = None
    
    for idx, span in sorted_spans:
        if idx in used_indices or idx == first_span_idx:
            continue
        gap = span.start - connector_end_pos
        if -2 <= gap <= 10:
            second_span = span
            second_span_idx = idx
            break
    
    if not second_span:
        return None
    
    # Parse both spans
    first_expr = _parse_span_to_scatex(first_span.text, settings)
    second_expr = _parse_span_to_scatex(second_span.text, settings)
    
    if not first_expr or not second_expr:
        return None
    
    # Propagate context
    second_expr = _propagate_context(first_expr, second_expr)
    first_expr, second_expr = _propagate_year_in_range(first_expr, second_expr)
    
    # Create Between expression
    between_expr = Between(start_interval=first_expr, end_interval=second_expr)
    
    full_text = text[signal.start:second_span.end]
    
    composite = CompositeSpan(
        text=full_text,
        start=signal.start,
        end=second_span.end,
        expression=between_expr,
        pattern_name="signal_merge:from_to_range"
    )
    
    return (composite, [first_span_idx, second_span_idx])


def _try_merge_signal_span(
    text: str,
    signal: Any,
    signal_idx: int,
    sorted_spans: List[Tuple[int, Any]],
    used_indices: set,
    scatex_type: type,
    settings: Any
) -> Optional[Tuple[CompositeSpan, int]]:
    """
    Try to merge a signal with an adjacent span.
    
    Pattern: "[signal] [span]" → scatex_type(span_expr)
    """
    # Find first unused span after signal within adjacency window
    for idx, span in sorted_spans:
        if idx in used_indices:
            continue
        
        gap = span.start - signal.end
        if 0 <= gap <= MAX_ADJACENCY_GAP:
            # Check that the gap contains only whitespace and simple words
            # (not another temporal expression)
            gap_text = text[signal.end:span.start]
            if _is_valid_gap(gap_text):
                # Parse the span
                span_expr = _parse_span_to_scatex(span.text, settings)
                if span_expr:
                    # Create composite expression
                    composite_expr = scatex_type(interval=span_expr)
                    
                    full_text = text[signal.start:span.end]
                    composite = CompositeSpan(
                        text=full_text,
                        start=signal.start,
                        end=span.end,
                        expression=composite_expr,
                        pattern_name=f"signal_merge:{scatex_type.__name__.lower()}"
                    )
                    return (composite, idx)
        
        # If span is beyond adjacency window, stop searching
        if span.start > signal.end + MAX_ADJACENCY_GAP:
            break
    
    return None


def _find_dash_ranges(
    text: str,
    sorted_spans: List[Tuple[int, Any]],
    used_indices: set,
    settings: Any
) -> List[Tuple[CompositeSpan, List[int]]]:
    """
    Find adjacent spans connected by dashes (e.g., "2008-2009").
    """
    results = []
    
    for i, (idx1, span1) in enumerate(sorted_spans):
        if idx1 in used_indices:
            continue
        
        # Look for next span
        for j in range(i + 1, len(sorted_spans)):
            idx2, span2 = sorted_spans[j]
            if idx2 in used_indices:
                continue
            
            # Check if there's a dash between them
            between_text = text[span1.end:span2.start]
            
            # Must be just a dash (with optional whitespace)
            if re.match(r'^\s*[-–—]\s*$', between_text):
                # Parse both spans
                expr1 = _parse_span_to_scatex(span1.text, settings)
                expr2 = _parse_span_to_scatex(span2.text, settings)
                
                if expr1 and expr2:
                    # Propagate context (month from first, year from either)
                    expr2 = _propagate_context(expr1, expr2)
                    expr1, expr2 = _propagate_year_in_range(expr1, expr2)
                    
                    # Verify types are compatible (after context propagation)
                    if type(expr1) == type(expr2) or _are_compatible_types(expr1, expr2):
                        between_expr = Between(start_interval=expr1, end_interval=expr2)
                        
                        full_text = text[span1.start:span2.end]
                        composite = CompositeSpan(
                            text=full_text,
                            start=span1.start,
                            end=span2.end,
                            expression=between_expr,
                            pattern_name="signal_merge:dash_range"
                        )
                        results.append((composite, [idx1, idx2]))
                        used_indices.add(idx1)
                        used_indices.add(idx2)
            
            # Only check immediately adjacent spans
            break
    
    return results


def _is_valid_gap(gap_text: str) -> bool:
    """
    Check if the gap between signal and span is valid.
    
    Valid gaps contain only whitespace, punctuation, and simple connecting words.
    """
    # Strip whitespace and punctuation
    import string
    stripped = gap_text.strip().lower()
    
    # Remove punctuation (commas, etc.)
    stripped = stripped.translate(str.maketrans('', '', string.punctuation))
    stripped = stripped.strip()
    
    # Empty or whitespace only is valid
    if not stripped:
        return True
    
    # Allow simple connecting words
    valid_connectors = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to'}
    words = stripped.split()
    
    return all(w in valid_connectors for w in words)


def _are_compatible_types(expr1: TemporalExpression, expr2: TemporalExpression) -> bool:
    """Check if two expressions are compatible for a range."""
    compatible_pairs = [
        (Year, Year),
        (Month, Month),
        (Day, Day),
        (Hour, Hour),
        (Minute, Minute),
    ]
    return (type(expr1), type(expr2)) in compatible_pairs


def _propagate_context(first_expr: TemporalExpression, second_expr: TemporalExpression) -> TemporalExpression:
    """
    Propagate context (month, year) from first expression to second if missing.
    
    Examples:
    - "October 6" and Day(7) → Day(7, month=10) (inherit month)
    - Month(1) and Month(2, year=2024) → both get year 2024
    
    Args:
        first_expr: The first (context-providing) expression
        second_expr: The second expression that may need context
        
    Returns:
        Updated second_expr with propagated context
    """
    # Case 1: second is Month with no year, first is Month with year
    if isinstance(second_expr, Month) and isinstance(first_expr, Month):
        if second_expr.year is None and first_expr.year is not None:
            return Month(month=second_expr.month, year=first_expr.year)
        if first_expr.year is None and second_expr.year is not None:
            # Propagate backwards - update first_expr
            # But we can't modify first_expr here, so we return second as-is
            # The caller should handle backward propagation separately
            pass
    
    # Case 2: second is Day with no month, first is Day with month
    if isinstance(second_expr, Day) and isinstance(first_expr, Day):
        if second_expr.month is None and first_expr.month is not None:
            return Day(
                day=second_expr.day,
                month=first_expr.month,
                year=second_expr.year if second_expr.year is not None else first_expr.year
            )
    
    # Case 3: second is Month parsed from bare number (became Month(7) for "7")
    # but first is a Day - the "7" should be a day, not July
    if isinstance(second_expr, Month) and isinstance(first_expr, Day):
        # If second_expr.month looks like it could be a day (1-31) and has no year
        if second_expr.year is None and 1 <= second_expr.month <= 31:
            # Re-interpret as a Day with same month as first
            return Day(
                day=second_expr.month,  # The "month" is actually a day number
                month=first_expr.month,
                year=first_expr.year
            )
    
    return second_expr


def _propagate_year_in_range(expr1: TemporalExpression, expr2: TemporalExpression) -> Tuple[TemporalExpression, TemporalExpression]:
    """
    Propagate year between two expressions in a range.
    
    If one has a year and the other doesn't, propagate to both.
    
    Examples:
    - Month(1) and Month(2, year=2024) → Month(1, year=2024) and Month(2, year=2024)
    - Day(6, month=10) and Day(7, month=10, year=2023) → both get year 2023
    """
    # For Months
    if isinstance(expr1, Month) and isinstance(expr2, Month):
        year = expr1.year or expr2.year
        if year:
            expr1 = Month(month=expr1.month, year=year)
            expr2 = Month(month=expr2.month, year=year)
    
    # For Days
    if isinstance(expr1, Day) and isinstance(expr2, Day):
        year = expr1.year or expr2.year
        if year:
            expr1 = Day(day=expr1.day, month=expr1.month, year=year)
            expr2 = Day(day=expr2.day, month=expr2.month, year=year)
    
    return expr1, expr2


def _parse_span_to_scatex(span_text: str, settings: Any) -> Optional[TemporalExpression]:
    """Parse a span's text to a SCATEX expression."""
    try:
        import timeparser
        result = timeparser.parse(span_text, languages=['en'])
        if result and not isinstance(result, Unknown):
            return result
    except Exception as e:
        logger.debug(f"Failed to parse span '{span_text}': {e}")
    return None


# =============================================================================
# Public API
# =============================================================================

def try_parse_composite(date_string: str, settings: Any) -> Optional[Tuple[TemporalExpression, str]]:
    """
    Public API to try parsing a composite pattern.
    
    Args:
        date_string: The date string to parse
        settings: Parser settings
        
    Returns:
        Tuple of (expression, period) if matched, None otherwise.
    """
    return composite_parser.try_parse(date_string, settings)

