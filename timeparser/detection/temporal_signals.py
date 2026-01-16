"""
Temporal Signal Detection Service

This module detects and classifies temporal discourse markers, connectives, and
signals that TEI2GO may miss. These include:

- Discourse markers: "Later", "Subsequently", "Meanwhile"
- Anaphoric references: "the day before", "by then", "at that point"
- Relative phrases: "the following week", "the previous year"
- Frequency markers: "often", "repeatedly", "from time to time"

Temporal signals indicate relationships between events/times without being
concrete temporal expressions themselves. They require context or antecedent
resolution to anchor to specific dates.

Output includes SCATE (Semantically Compositional Annotation of Temporal 
Expressions) components for easy conversion to full SCATE expressions.

Based on TimeML annotation guidelines and Allen's Interval Algebra.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set, Union

# Import SCATE types from parent package (relative import within timeparser)
from ..scatex import (
    Period, Direction, Unit, 
    DayOfWeek, DayOfWeekType,
    Repeating,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class TemporalRelation(Enum):
    """
    Temporal relation types based on TimeML and Allen's Interval Algebra.
    
    These represent the relationship that a signal indicates between events.
    """
    BEFORE = "before"                       # X ends before Y begins
    AFTER = "after"                         # X begins after Y ends
    IMMEDIATELY_BEFORE = "immediately_before"  # X ends exactly when Y begins
    IMMEDIATELY_AFTER = "immediately_after"    # X begins exactly when Y ends
    SIMULTANEOUS = "simultaneous"           # X and Y occur at the same time
    OVERLAP = "overlap"                     # X and Y partially overlap
    DURATION = "duration"                   # Indicates temporal extent
    BEGINNING = "beginning"                 # Marks start of interval
    ENDING = "ending"                       # Marks end of interval
    FREQUENCY = "frequency"                 # Indicates recurrence pattern
    SEQUENCE = "sequence"                   # General ordering marker
    ANAPHORIC = "anaphoric"                 # Requires antecedent resolution


class SignalCategory(Enum):
    """Grammatical/functional category of the temporal signal."""
    ADVERB = "adverb"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    PHRASE = "phrase"
    DISCOURSE_MARKER = "discourse_marker"   # Sentence-initial markers
    ANAPHORIC_REFERENCE = "anaphoric_reference"
    FREQUENCY_ADVERB = "frequency_adverb"
    MODIFIER = "modifier"


@dataclass
class TemporalSignal:
    """A detected temporal signal/discourse marker."""
    text: str                               # The matched text
    start: int                              # Start character offset
    end: int                                # End character offset
    relation: TemporalRelation              # Classified relation type
    category: SignalCategory                # Grammatical category
    sentence: str                           # Containing sentence
    sentence_start: int                     # Start of sentence in text
    is_anaphoric: bool = False              # Requires antecedent resolution
    is_sentence_initial: bool = False       # Appears at start of sentence
    # SCATE components for structured output
    scate_period: Optional[Period] = None   # e.g., Period(unit=DAY, value=2) for "two days"
    scate_direction: Optional[Direction] = None  # BEFORE or AFTER
    scate_repeating: Optional[Union[DayOfWeek, Repeating]] = None  # For weekdays/patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "relation": self.relation.value,
            "category": self.category.value,
            "sentence": self.sentence,
            "sentence_start": self.sentence_start,
            "is_anaphoric": self.is_anaphoric,
            "is_sentence_initial": self.is_sentence_initial,
        }
        # Add SCATE components if present
        if self.scate_period:
            result["scate_period"] = repr(self.scate_period)
        if self.scate_direction:
            result["scate_direction"] = self.scate_direction.value
        if self.scate_repeating:
            result["scate_repeating"] = repr(self.scate_repeating)
        return result


# =============================================================================
# Signal Lexicon
# =============================================================================

# Each entry: (pattern, relation, category, is_anaphoric)
# Patterns are matched with word boundaries automatically

SIGNAL_LEXICON: Dict[TemporalRelation, Dict[SignalCategory, List[Tuple[str, bool]]]] = {
    
    # =========================================================================
    # BEFORE - indicates precedence relationship
    # =========================================================================
    TemporalRelation.BEFORE: {
        SignalCategory.ADVERB: [
            ("before", False),
            ("beforehand", False),
            ("earlier", False),
            ("previously", False),
            ("formerly", False),
            ("prior", False),
            ("already", False),
            ("first", False),
            ("initially", False),
            ("originally", False),
            ("hitherto", False),
            ("heretofore", False),
            ("theretofore", False),
            ("once", False),  # "once upon a time", "once he arrived"
            ("recently", False),
            ("lately", False),
            ("latterly", False),
        ],
        SignalCategory.MODIFIER: [
            ("former", False),
            ("previous", False),
            ("preliminary", False),
            ("long-standing", False),
        ],
        SignalCategory.PREPOSITION: [
            ("before", False),
            ("prior to", False),
            ("previous to", False),
            ("ahead of", False),
            ("in advance of", False),
            ("preceding", False),
        ],
        SignalCategory.CONJUNCTION: [
            ("before", False),
            ("until", False),
            ("by the time", False),
            ("ere", False),  # archaic
        ],
        SignalCategory.PHRASE: [
            ("in the past", False),
            ("up to that point", True),
            ("up until then", True),
            ("in earlier times", False),
            ("in former times", False),
            ("back then", True),
            ("back in the day", False),
            # Additional past idioms
            ("the other day", False),
            ("ages ago", False),
            ("way back", False),
            ("way back when", False),
            ("long ago", False),
            ("in the old days", False),
            ("in times past", False),
            ("in years past", False),
            ("in days gone by", False),
            ("once upon a time", False),
            ("at one time", False),
            ("in retrospect", False),
            ("in hindsight", False),
            ("looking back", False),
            ("long since", False),
        ],
        SignalCategory.ANAPHORIC_REFERENCE: [
            ("the day before", True),
            ("the week before", True),
            ("the month before", True),
            ("the year before", True),
            ("the night before", True),
            ("the evening before", True),
            ("the morning before", True),
            ("the previous day", True),
            ("the previous week", True),
            ("the previous month", True),
            ("the previous year", True),
            ("the previous night", True),
            ("the prior day", True),
            ("the prior week", True),
            ("the prior month", True),
            ("the prior year", True),
            ("the preceding day", True),
            ("the preceding week", True),
            ("the preceding month", True),
            ("the preceding year", True),
            ("the day before that", True),
            ("the week before that", True),
            ("the year before that", True),
            ("a day earlier", True),
            ("a week earlier", True),
            ("a month earlier", True),
            ("a year earlier", True),
            ("days earlier", True),
            ("weeks earlier", True),
            ("months earlier", True),
            ("years earlier", True),
            ("some time before", True),
            ("shortly before", True),
            ("just before", True),
            ("right before", True),
            ("long before", True),
            ("not long before", True),
        ],
        SignalCategory.DISCOURSE_MARKER: [
            ("Previously", False),
            ("Earlier", False),
            ("Before that", True),
            ("Before this", True),
            ("Prior to that", True),
            ("Prior to this", True),
        ],
    },
    
    # =========================================================================
    # AFTER - indicates succession relationship
    # =========================================================================
    TemporalRelation.AFTER: {
        SignalCategory.ADVERB: [
            ("after", False),
            ("afterward", False),
            ("afterwards", False),
            ("later", False),
            ("subsequently", False),
            ("then", False),
            ("next", False),
            ("thereafter", False),
            ("thenceforth", False),
            ("henceforth", False),
            ("eventually", False),
            ("finally", False),
            ("ultimately", False),
            ("lastly", False),
            ("soon", False),
            ("shortly", False),
            ("presently", False),
            ("hereafter", False),
            ("forthcoming", False),
            ("impending", False),
            ("incoming", False),
        ],
        SignalCategory.MODIFIER: [
            ("subsequent", False),
        ],
        SignalCategory.PREPOSITION: [
            ("after", False),
            ("following", False),
            ("subsequent to", False),
            ("in the wake of", False),
            ("in the aftermath of", False),
            ("in the immediate aftermath of", False),
            ("immediately after", False),
            ("directly after", False),
            ("shortly after", False),
            ("right after", False),
            ("post", False),
            ("followed by", False),
        ],
        SignalCategory.CONJUNCTION: [
            ("after", False),
            ("once", False),
            ("when", False),
            ("as soon as", False),
            ("the moment", False),
            ("now that", False),
        ],
        SignalCategory.PHRASE: [
            ("from then on", True),
            ("from that point", True),
            ("from that point on", True),
            ("from that moment", True),
            ("from that moment on", True),
            ("from that time", True),
            ("from that time on", True),
            ("from this point", True),
            ("from this moment", True),
            ("in the future", False),
            ("going forward", False),
            ("moving forward", False),
            ("in the end", False),
            ("at the end", False),
            ("at last", False),
            ("in time", False),
            ("with time", False),
            ("over time", False),
            ("as time went on", False),
            ("as time passed", False),
            # Additional future idioms
            ("in days to come", False),
            ("in weeks to come", False),
            ("in months to come", False),
            ("in years to come", False),
            ("in the days ahead", False),
            ("in the weeks ahead", False),
            ("in the months ahead", False),
            ("in the years ahead", False),
            ("down the road", False),
            ("down the line", False),
            ("further down the road", False),
            ("further down the line", False),
            ("sooner or later", False),
            ("in the foreseeable future", False),
            ("for the foreseeable future", False),
            ("someday", False),
            ("looking ahead", False),
            ("from here on", False),
            ("from here on out", False),
            ("from now on", False),
            ("in due time", False),
            ("in due course", False),
            ("in good time", False),
        ],
        SignalCategory.ANAPHORIC_REFERENCE: [
            ("the day after", True),
            ("the week after", True),
            ("the month after", True),
            ("the year after", True),
            ("the night after", True),
            ("the morning after", True),
            ("the evening after", True),
            ("the following day", True),
            ("the following week", True),
            ("the following month", True),
            ("the following year", True),
            ("the following night", True),
            ("the following morning", True),
            ("the next day", True),
            ("the next week", True),
            ("the next month", True),
            ("the next year", True),
            ("the next morning", True),
            ("the next night", True),
            ("the subsequent day", True),
            ("the subsequent week", True),
            ("the subsequent month", True),
            ("the subsequent year", True),
            ("the ensuing day", True),
            ("the ensuing week", True),
            ("the ensuing month", True),
            ("the ensuing year", True),
            ("the day after that", True),
            ("the week after that", True),
            ("the year after that", True),
            ("a day later", True),
            ("a week later", True),
            ("a month later", True),
            ("a year later", True),
            ("days later", True),
            ("weeks later", True),
            ("months later", True),
            ("years later", True),
            ("some time later", True),
            ("some time after", True),
            ("shortly after", True),
            ("shortly thereafter", True),
            ("soon after", True),
            ("soon thereafter", True),
            ("just after", True),
            ("right after", True),
            ("long after", True),
            ("not long after", True),
        ],
        SignalCategory.DISCOURSE_MARKER: [
            ("Later", False),
            ("Afterward", False),
            ("Afterwards", False),
            ("Subsequently", False),
            ("Then", False),
            ("Next", False),
            ("Thereafter", False),
            ("Eventually", False),
            ("Finally", False),
            ("Ultimately", False),
            ("In the end", False),
            ("At last", False),
            ("After that", True),
            ("After this", True),
            ("Following that", True),
            ("Following this", True),
        ],
    },
    
    # =========================================================================
    # IMMEDIATELY_AFTER - marks immediate succession
    # =========================================================================
    TemporalRelation.IMMEDIATELY_AFTER: {
        SignalCategory.ADVERB: [
            ("immediately", False),
            ("instantly", False),
            ("straightaway", False),
            ("forthwith", False),
            ("promptly", False),
            ("directly", False),
            ("thereupon", False),
            ("whereupon", False),
            ("quickly", False),
            ("readily", False),
        ],
        SignalCategory.PREPOSITION: [
            ("immediately upon", False),
        ],
        SignalCategory.PHRASE: [
            ("right away", False),
            ("at once", False),
            ("right then", True),
            ("right after", True),
            ("just after", True),
            ("straight after", True),
            ("the moment after", True),
            ("the instant", True),
            ("no sooner", False),
            ("in no time", False),
            ("without delay", False),
            ("on the spot", False),
            ("then and there", False),
            ("there and then", False),
            ("on the double", False),
            ("just as soon as", False),
            ("immediately followed by", False),
        ],
        SignalCategory.DISCOURSE_MARKER: [
            ("Immediately", False),
            ("Instantly", False),
            ("Straightaway", False),
            ("At once", False),
            ("Right away", False),
        ],
    },
    
    # =========================================================================
    # SIMULTANEOUS - marks co-occurrence
    # =========================================================================
    TemporalRelation.SIMULTANEOUS: {
        SignalCategory.ADVERB: [
            ("simultaneously", False),
            ("concurrently", False),
            ("meanwhile", False),
            ("meantime", False),
            ("together", False),
            ("concomitantly", False),
            ("contemporaneously", False),
            ("jointly", False),
        ],
        SignalCategory.PREPOSITION: [
            ("during", False),
            ("amid", False),
            ("amidst", False),
            ("in the midst of", False),
            ("in the course of", False),
            ("over", False),
            ("through", False),
            ("throughout", False),
            ("around", False),  # approximate time
            ("about", False),   # approximate time
            ("circa", False),   # approximate time
            ("between", False), # interval
        ],
        SignalCategory.CONJUNCTION: [
            ("while", False),
            ("whilst", False),
            ("as", False),
            ("when", False),
            ("whenever", False),
            ("at the same time as", False),
            ("even as", False),
        ],
        SignalCategory.PHRASE: [
            ("at the same time", False),
            ("at that time", True),
            ("at this time", True),
            ("at the moment", False),
            ("at the very moment", False),
            ("in the meantime", False),
            ("in the meanwhile", False),
            ("all the while", False),
            ("during this time", True),
            ("during that time", True),
            ("during that period", True),
            ("during this period", True),
            ("at that point", True),
            ("at this point", True),
            ("at that moment", True),
            ("at this moment", True),
            ("in parallel", False),
            ("in conjunction", False),
            ("in tandem", False),
            ("hand in hand", False),
            ("side by side", False),
            ("all along", False),
        ],
        SignalCategory.ANAPHORIC_REFERENCE: [
            ("by then", True),
            ("by that time", True),
            ("by this time", True),
            ("by that point", True),
            ("by this point", True),
        ],
        SignalCategory.DISCOURSE_MARKER: [
            ("Meanwhile", False),
            ("In the meantime", False),
            ("At the same time", False),
            ("Simultaneously", False),
            ("Concurrently", False),
            ("During this time", True),
            ("All the while", False),
        ],
    },
    
    # =========================================================================
    # DURATION - indicates temporal extent
    # =========================================================================
    TemporalRelation.DURATION: {
        SignalCategory.ADVERB: [
            ("still", False),
            ("yet", False),
            ("always", False),
            ("continuously", False),
            ("constantly", False),
            ("perpetually", False),
            ("forever", False),
            ("indefinitely", False),
            ("temporarily", False),
            ("briefly", False),
            ("momentarily", False),
            ("long", False),
            ("longer", False),
            ("tentatively", False),
        ],
        SignalCategory.PREPOSITION: [
            ("for", False),
            ("during", False),
            ("throughout", False),
            ("over", False),
            ("across", False),
            ("within", False),
            ("pending", False),
        ],
        SignalCategory.CONJUNCTION: [
            ("while", False),
            ("as long as", False),
            ("until", False),
            ("till", False),
            ("up till", False),
            ("just as long as", False),
        ],
        SignalCategory.PHRASE: [
            ("the whole time", False),
            ("the entire time", False),
            ("for the duration", False),
            ("for a while", False),
            ("for some time", False),
            ("for a time", False),
            ("up to now", False),
            ("so far", False),
            ("thus far", False),
            ("to date", False),
            ("to this day", False),
            ("ever since", True),
            ("on end", False),  # "for hours on end"
            ("at a stretch", False),
            ("nonstop", False),
            ("non-stop", False),
            ("around the clock", False),
            ("day and night", False),
            ("night and day", False),
            ("24/7", False),
            ("round the clock", False),
            ("for the time being", False),
            ("for now", False),
            ("for good", False),  # permanent
            ("in the interim", False),
            ("in the intervening time", False),
        ],
    },
    
    # =========================================================================
    # BEGINNING - marks inception
    # =========================================================================
    TemporalRelation.BEGINNING: {
        SignalCategory.ADVERB: [
            ("initially", False),
            ("originally", False),
            ("first", False),
            ("firstly", False),
        ],
        SignalCategory.PREPOSITION: [
            ("from", False),
            ("since", False),
            ("starting", False),
            ("beginning", False),
            ("as of", False),
            ("effective", False),
        ],
        SignalCategory.PHRASE: [
            ("at first", False),
            ("in the beginning", False),
            ("at the start", False),
            ("at the outset", False),
            ("at the onset", False),
            ("to begin with", False),
            ("to start with", False),
            ("from the start", False),
            ("from the beginning", False),
            ("from the outset", False),
            ("from day one", False),
            ("from the get-go", False),
            ("right from the start", False),
            ("at the dawn of", False),
            ("in the early stages", False),
            ("in the early days", False),
            ("as early as", False),
            ("right off the bat", False),
            ("straight off", False),
            ("from the word go", False),
        ],
        SignalCategory.MODIFIER: [
            ("early", False),
            ("initial", False),
        ],
        SignalCategory.DISCOURSE_MARKER: [
            ("Initially", False),
            ("Originally", False),
            ("First", False),
            ("At first", False),
            ("In the beginning", False),
            ("To begin with", False),
            ("To start with", False),
        ],
    },
    
    # =========================================================================
    # ENDING - marks conclusion
    # =========================================================================
    TemporalRelation.ENDING: {
        SignalCategory.ADVERB: [
            ("finally", False),
            ("eventually", False),
            ("ultimately", False),
            ("lastly", False),
            ("conclusively", False),
        ],
        SignalCategory.PREPOSITION: [
            ("until", False),
            ("till", False),
            ("by", False),
            ("up to", False),
            ("through", False),
        ],
        SignalCategory.PHRASE: [
            ("in the end", False),
            ("at the end", False),
            ("at last", False),
            ("at long last", False),
            ("at length", False),
            ("in conclusion", False),
            ("to conclude", False),
            ("by the end of", False),
            ("at the close of", False),
            ("in the final analysis", False),
            ("in the long run", False),
            ("when all was said and done", False),
            ("when all is said and done", False),
        ],
        SignalCategory.DISCOURSE_MARKER: [
            ("Finally", False),
            ("Eventually", False),
            ("Ultimately", False),
            ("Lastly", False),
            ("In the end", False),
            ("At last", False),
            ("In conclusion", False),
        ],
    },
    
    # =========================================================================
    # FREQUENCY - indicates recurrence
    # =========================================================================
    TemporalRelation.FREQUENCY: {
        SignalCategory.FREQUENCY_ADVERB: [
            ("always", False),
            ("never", False),
            ("often", False),
            ("frequently", False),
            ("usually", False),
            ("sometimes", False),
            ("occasionally", False),
            ("rarely", False),
            ("seldom", False),
            ("regularly", False),
            ("periodically", False),
            ("repeatedly", False),
            ("continually", False),
            ("daily", False),
            ("weekly", False),
            ("monthly", False),
            ("yearly", False),
            ("annually", False),
            ("hourly", False),
            ("nightly", False),
            ("biweekly", False),
            ("fortnightly", False),
            ("quarterly", False),
            ("again", False),
            ("once more", False),
            ("twice", False),
            ("thrice", False),
            ("habitually", False),
            ("customarily", False),
            ("routinely", False),
            ("typically", False),
            ("normally", False),
            ("generally", False),
            ("commonly", False),
            ("infrequently", False),
            ("sporadically", False),
            ("intermittently", False),
        ],
        SignalCategory.PHRASE: [
            ("from time to time", False),
            ("now and then", False),
            ("now and again", False),
            ("every now and then", False),
            ("every now and again", False),
            ("once in a while", False),
            ("on occasion", False),
            ("at times", False),
            ("on and off", False),
            ("off and on", False),
            ("time after time", False),
            ("time and again", False),
            ("time and time again", False),
            ("over and over", False),
            ("over and over again", False),
            ("again and again", False),
            ("every day", False),
            ("each week", False),
            ("per month", False),
            ("every so often", False),
            ("more often than not", False),
            ("once a day", False),
            ("once a week", False),
            ("twice a day", False),
            ("twice a week", False),
            ("every other day", False),
            ("every other week", False),
            ("day after day", False),
            ("week after week", False),
            ("year after year", False),
            ("on a daily basis", False),
            ("on a regular basis", False),
        ],
    },
    
    # =========================================================================
    # SEQUENCE - general ordering (when specific relation unclear)
    # =========================================================================
    TemporalRelation.SEQUENCE: {
        SignalCategory.ADVERB: [
            ("second", False),
            ("secondly", False),
            ("third", False),
            ("thirdly", False),
            ("fourth", False),
            ("fourthly", False),
            ("fifth", False),
            ("fifthly", False),
            ("last", False),
            ("penultimately", False),
            ("consecutively", False),
        ],
        SignalCategory.PHRASE: [
            ("in turn", False),
            ("in sequence", False),
            ("in order", False),
            ("in succession", False),
            ("in a row", False),
            ("back-to-back", False),
            ("one by one", False),
            ("step by step", False),
            ("one after another", False),
            ("one after the other", False),
        ],
        SignalCategory.DISCOURSE_MARKER: [
            ("Second", False),
            ("Secondly", False),
            ("Third", False),
            ("Thirdly", False),
            ("Fourth", False),
            ("Fifth", False),
            ("Last", False),
            ("Lastly", False),
        ],
    },
    
    # =========================================================================
    # ANAPHORIC - pure anaphoric references (relation depends on context)
    # =========================================================================
    TemporalRelation.ANAPHORIC: {
        SignalCategory.ANAPHORIC_REFERENCE: [
            ("then", True),
            ("now", True),
            ("that time", True),
            ("this time", True),
            ("that moment", True),
            ("this moment", True),
            ("that day", True),
            ("this day", True),
            ("that year", True),
            ("this year", True),
            ("that week", True),
            ("this week", True),
            ("that month", True),
            ("this month", True),
            ("that period", True),
            ("this period", True),
            ("that era", True),
            ("this era", True),
            ("that point", True),
            ("this point", True),
            ("that stage", True),
            ("this stage", True),
            ("that juncture", True),
            ("this juncture", True),
            ("the time", True),
            ("the moment", True),
            ("the period", True),
            ("since then", True),
            ("until then", True),
            ("before then", True),
            ("after then", True),
            ("around then", True),
            ("about then", True),
        ],
    },
}


# =============================================================================
# Template-based Pattern Matching
# =============================================================================

# Time units (singular forms - plurals are handled dynamically)
TIME_UNITS = [
    "second", "minute", "hour", "day", "week", "fortnight", "month",
    "quarter", "year", "decade", "century", "millennium",
    "moment", "instant",
    "morning", "afternoon", "evening", "night",
    "spring", "summer", "fall", "autumn", "winter",
]

# Word-to-number mapping for quantifiers
WORD_TO_NUMBER: Dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "the": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "couple": 2, "few": 3, "several": 3, "some": 1, "many": 5,
    "half": 1,  # "half a day" = 1 (unit handles the half)
}

# Direction markers and their signs
BEFORE_MARKERS = ["before", "earlier", "prior", "ago", "previously", "preceding"]
AFTER_MARKERS = ["after", "later", "following", "hence", "subsequent", "thereafter"]

# Map string units to SCATE Unit enum
UNIT_TO_SCATE: Dict[str, Unit] = {
    "second": Unit.SECOND,
    "minute": Unit.MINUTE,
    "hour": Unit.HOUR,
    "day": Unit.DAY,
    "week": Unit.WEEK,
    "fortnight": Unit.WEEK,  # fortnight = 2 weeks, handled in value
    "month": Unit.MONTH,
    "quarter": Unit.QUARTER,
    "year": Unit.YEAR,
    "decade": Unit.DECADE,
    "century": Unit.CENTURY,
    "millennium": Unit.CENTURY,  # millennium = 10 centuries, handled in value
    # These don't map cleanly to SCATE units, will use approximations
    "moment": Unit.SECOND,
    "instant": Unit.SECOND,
    "morning": Unit.HOUR,
    "afternoon": Unit.HOUR,
    "evening": Unit.HOUR,
    "night": Unit.HOUR,
    "spring": Unit.MONTH,
    "summer": Unit.MONTH,
    "fall": Unit.MONTH,
    "autumn": Unit.MONTH,
    "winter": Unit.MONTH,
}

# Special unit multipliers (fortnight = 2 weeks, millennium = 10 centuries)
UNIT_MULTIPLIERS: Dict[str, int] = {
    "fortnight": 2,
    "millennium": 10,
}

# Days of week for day-of-week templates
DAYS_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Map day names to SCATE DayOfWeekType
DAY_TO_SCATE: Dict[str, DayOfWeekType] = {
    "monday": DayOfWeekType.MONDAY,
    "tuesday": DayOfWeekType.TUESDAY,
    "wednesday": DayOfWeekType.WEDNESDAY,
    "thursday": DayOfWeekType.THURSDAY,
    "friday": DayOfWeekType.FRIDAY,
    "saturday": DayOfWeekType.SATURDAY,
    "sunday": DayOfWeekType.SUNDAY,
}

# Tens place for compound numbers (twenty-three -> 23)
TENS_PLACE: Dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# Build regex components
_UNIT_PATTERN = r"(?P<unit>" + "|".join(TIME_UNITS) + r")s?"  # Allow plural
_QUANT_WORDS = "|".join(WORD_TO_NUMBER.keys())
# Include compound numbers like "twenty-three" in quantifier pattern
_TENS_WORDS = "|".join(TENS_PLACE.keys())
_ONES_WORDS = "|".join(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
_COMPOUND_PATTERN = rf"(?:{_TENS_WORDS})-(?:{_ONES_WORDS})"
_QUANT_PATTERN = r"(?P<quant>" + _COMPOUND_PATTERN + r"|" + _QUANT_WORDS + r"|\d+)"
_BEFORE_PATTERN = r"(?P<dir>" + "|".join(BEFORE_MARKERS) + r")"
_AFTER_PATTERN = r"(?P<dir>" + "|".join(AFTER_MARKERS) + r")"


@dataclass
class TemporalTemplate:
    """A template pattern for matching temporal expressions with offsets."""
    pattern: re.Pattern
    relation: TemporalRelation
    default_offset: Optional[int] = None  # For patterns without explicit quantifier
    fixed_unit: Optional[Unit] = None  # Override unit for special patterns (e.g., fractions)


# Relations that map to SCATE Direction
_BEFORE_RELATIONS = {TemporalRelation.BEFORE, TemporalRelation.IMMEDIATELY_BEFORE}
_AFTER_RELATIONS = {TemporalRelation.AFTER, TemporalRelation.IMMEDIATELY_AFTER}


def _relation_to_direction(relation: TemporalRelation) -> Optional[Direction]:
    """Convert a TemporalRelation to a SCATE Direction."""
    if relation in _BEFORE_RELATIONS:
        return Direction.BEFORE
    elif relation in _AFTER_RELATIONS:
        return Direction.AFTER
    return None


# Template patterns - order matters (more specific first)
TEMPORAL_TEMPLATES: List[TemporalTemplate] = []


def _build_templates() -> List[TemporalTemplate]:
    """
    Build and compile all template patterns.
    
    IMPORTANT: Order matters! More specific patterns must come FIRST
    to avoid being shadowed by general patterns.
    """
    templates = []
    
    _DOW_PATTERN = r"(?P<day>" + "|".join(DAYS_OF_WEEK) + r")"
    
    # ==========================================================================
    # MOST SPECIFIC: Fraction patterns (must match before general hour patterns)
    # ==========================================================================
    
    # half an hour before/after (= 30 minutes)
    templates.append(TemporalTemplate(
        pattern=re.compile(
            r"\bhalf\s+an?\s+hour\s+(?:before|earlier|prior|ago)\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=30,
        fixed_unit=Unit.MINUTE,
    ))
    
    templates.append(TemporalTemplate(
        pattern=re.compile(
            r"\bhalf\s+an?\s+hour\s+(?:after|later)\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=30,
        fixed_unit=Unit.MINUTE,
    ))
    
    # quarter of an hour (= 15 minutes)
    templates.append(TemporalTemplate(
        pattern=re.compile(
            r"\b(?:a\s+)?quarter\s+(?:of\s+an?\s+)?hour\s+(?:before|earlier|prior|ago)\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=15,
        fixed_unit=Unit.MINUTE,
    ))
    
    templates.append(TemporalTemplate(
        pattern=re.compile(
            r"\b(?:a\s+)?quarter\s+(?:of\s+an?\s+)?hour\s+(?:after|later)\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=15,
        fixed_unit=Unit.MINUTE,
    ))
    
    # ==========================================================================
    # SPECIFIC: Intensified distance (must match before "the [unit] before")
    # ==========================================================================
    
    # the [unit] before last (= 2 units before)
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+{_UNIT_PATTERN}\s+before\s+last\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=2,
    ))
    
    # the [unit] after next (= 2 units after)
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+{_UNIT_PATTERN}\s+after\s+next\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=2,
    ))
    
    # ==========================================================================
    # SPECIFIC: Day-of-week patterns
    # ==========================================================================
    
    # the [day] before/after
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+{_DOW_PATTERN}\s+(?:before|prior)\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=1,
    ))
    
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+{_DOW_PATTERN}\s+(?:after|following)\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=1,
    ))
    
    # the following/previous/next [day]
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+(?:previous|prior|preceding|last)\s+{_DOW_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=1,
    ))
    
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+(?:following|next|subsequent)\s+{_DOW_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=1,
    ))
    
    # ==========================================================================
    # SPECIFIC: "or so" patterns (longer than base patterns)
    # ==========================================================================
    
    # [quant] [unit](s) or so before/earlier
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\b{_QUANT_PATTERN}\s+{_UNIT_PATTERN}\s+or\s+so\s+{_BEFORE_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
    ))
    
    # [quant] [unit](s) or so after/later
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\b{_QUANT_PATTERN}\s+{_UNIT_PATTERN}\s+or\s+so\s+{_AFTER_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
    ))
    
    # ==========================================================================
    # GENERAL: Basic [quant] [unit] before/after patterns
    # ==========================================================================
    
    # [quant] [unit](s) before/earlier/prior/ago
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\b{_QUANT_PATTERN}\s+{_UNIT_PATTERN}\s+{_BEFORE_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
    ))
    
    # [quant] [unit](s) after/later/following/hence
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\b{_QUANT_PATTERN}\s+{_UNIT_PATTERN}\s+{_AFTER_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
    ))
    
    # ==========================================================================
    # GENERAL: "the previous/following [unit]" patterns
    # ==========================================================================
    
    # the previous/preceding/prior [unit]
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+(?:previous|preceding|prior)\s+{_UNIT_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=1,
    ))
    
    # the following/next/subsequent/ensuing [unit]
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\bthe\s+(?:following|next|subsequent|ensuing)\s+{_UNIT_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=1,
    ))
    
    # ==========================================================================
    # GENERAL: Vague modifier patterns (no numeric offset)
    # ==========================================================================
    
    # shortly/just/right/long/soon before
    templates.append(TemporalTemplate(
        pattern=re.compile(
            r"\b(?:shortly|just|right|long|not\s+long|soon|immediately)\s+before\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=None,
    ))
    
    # shortly/just/right/long/soon after
    templates.append(TemporalTemplate(
        pattern=re.compile(
            r"\b(?:shortly|just|right|long|not\s+long|soon|immediately)\s+after\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=None,
    ))
    
    # ==========================================================================
    # LEAST SPECIFIC: Bare [unit] before/after (without quantifier = 1)
    # ==========================================================================
    
    # [unit](s) before/earlier
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\b{_UNIT_PATTERN}\s+{_BEFORE_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.BEFORE,
        default_offset=1,
    ))
    
    # [unit](s) after/later
    templates.append(TemporalTemplate(
        pattern=re.compile(
            rf"\b{_UNIT_PATTERN}\s+{_AFTER_PATTERN}\b",
            re.IGNORECASE
        ),
        relation=TemporalRelation.AFTER,
        default_offset=1,
    ))
    
    return templates


# Initialize templates at module load
TEMPORAL_TEMPLATES = _build_templates()


def _parse_quantifier(quant_str: str) -> Optional[int]:
    """Parse a quantifier string to an integer, including compound numbers."""
    if not quant_str:
        return None
    
    quant_lower = quant_str.lower().strip()
    
    # Try word mapping first
    if quant_lower in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[quant_lower]
    
    # Try compound number (twenty-three, forty-five, etc.)
    if "-" in quant_lower:
        parts = quant_lower.split("-")
        if len(parts) == 2:
            tens = TENS_PLACE.get(parts[0], 0)
            ones = WORD_TO_NUMBER.get(parts[1], 0)
            if tens > 0:
                return tens + ones
    
    # Try parsing as digit
    try:
        return int(quant_str)
    except ValueError:
        return None


def _normalize_unit(unit_str: str) -> str:
    """Normalize a unit string to singular lowercase form."""
    if not unit_str:
        return ""
    
    unit = unit_str.lower()
    
    # Remove plural 's' if present
    if unit.endswith("s") and unit[:-1] in TIME_UNITS:
        unit = unit[:-1]
    
    # Handle special cases
    if unit == "autumns":
        unit = "autumn"
    
    return unit


def _create_scate_period(unit_str: str, value: int) -> Optional[Period]:
    """
    Create a SCATE Period from a unit string and value.
    
    Handles special cases like fortnight (= 2 weeks).
    """
    if not unit_str:
        return None
    
    unit_lower = unit_str.lower()
    
    # Get SCATE unit
    scate_unit = UNIT_TO_SCATE.get(unit_lower)
    if not scate_unit:
        return None
    
    # Apply multiplier (e.g., fortnight = 2 weeks)
    multiplier = UNIT_MULTIPLIERS.get(unit_lower, 1)
    final_value = value * multiplier
    
    return Period(unit=scate_unit, value=final_value)


def _extract_template_matches(
    text: str,
    sentences: List[Tuple[int, int, str]],
    seen_ranges: Set[Tuple[int, int]]
) -> List[TemporalSignal]:
    """
    Extract temporal signals using template patterns.
    
    Args:
        text: The input text
        sentences: Pre-segmented sentences
        seen_ranges: Already matched ranges (to avoid overlap)
        
    Returns:
        List of TemporalSignal objects from template matches with SCATE components
    """
    signals = []
    
    for template in TEMPORAL_TEMPLATES:
        for match in template.pattern.finditer(text):
            start, end = match.span()
            
            # Skip if overlapping with existing match
            overlaps = False
            for seen_start, seen_end in seen_ranges:
                if start < seen_end and end > seen_start:
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            # Extract components from named groups
            groups = match.groupdict()
            quant_str = groups.get("quant")
            unit_str = groups.get("unit")
            day_str = groups.get("day")  # For day-of-week templates
            
            # Parse quantifier
            quant = _parse_quantifier(quant_str) if quant_str else template.default_offset
            
            # Normalize unit
            unit = _normalize_unit(unit_str) if unit_str else None
            
            # Build SCATE components
            scate_period: Optional[Period] = None
            scate_repeating: Optional[Union[DayOfWeek, Repeating]] = None
            
            # Set direction based on relation type
            scate_direction = _relation_to_direction(template.relation)
            
            # Handle day-of-week matches
            if day_str:
                day_lower = day_str.lower()
                if day_lower in DAY_TO_SCATE:
                    scate_repeating = DayOfWeek(type=DAY_TO_SCATE[day_lower])
            # Handle fixed unit patterns (fractions like "half an hour")
            elif template.fixed_unit and quant is not None:
                scate_period = Period(unit=template.fixed_unit, value=quant)
            # Handle unit-based matches
            elif unit and quant is not None:
                scate_period = _create_scate_period(unit, quant)
            
            # Get sentence context
            sentence, sent_start, is_at_sent_start = _get_sentence_for_position(
                text, start, sentences
            )
            
            signals.append(TemporalSignal(
                text=match.group(),
                start=start,
                end=end,
                relation=template.relation,
                category=SignalCategory.PHRASE,
                sentence=sentence,
                sentence_start=sent_start,
                is_anaphoric=True,  # Template matches are typically anaphoric
                is_sentence_initial=is_at_sent_start,
                scate_period=scate_period,
                scate_direction=scate_direction,
                scate_repeating=scate_repeating,
            ))
            seen_ranges.add((start, end))
    
    return signals


# =============================================================================
# Pattern Compilation
# =============================================================================

@dataclass
class CompiledPattern:
    """A compiled regex pattern with metadata."""
    pattern: re.Pattern
    text: str
    relation: TemporalRelation
    category: SignalCategory
    is_anaphoric: bool
    is_discourse_marker: bool


# Compile all patterns at module load for efficiency
_compiled_patterns: List[CompiledPattern] = []


def _compile_patterns() -> List[CompiledPattern]:
    """Compile all signal patterns into regex patterns."""
    patterns = []
    
    for relation, categories in SIGNAL_LEXICON.items():
        for category, entries in categories.items():
            is_discourse_marker = category == SignalCategory.DISCOURSE_MARKER
            
            for text, is_anaphoric in entries:
                # Escape special regex characters
                escaped = re.escape(text)
                
                if is_discourse_marker:
                    # Discourse markers appear at sentence start
                    # Match at start of string or after sentence boundary
                    pattern_str = rf'(?:^|(?<=[.!?]\s))({escaped})(?=[,\s]|$)'
                    flags = re.MULTILINE
                else:
                    # Regular patterns with word boundaries
                    pattern_str = rf'\b({escaped})\b'
                    flags = re.IGNORECASE
                
                try:
                    compiled = re.compile(pattern_str, flags)
                    patterns.append(CompiledPattern(
                        pattern=compiled,
                        text=text,
                        relation=relation,
                        category=category,
                        is_anaphoric=is_anaphoric,
                        is_discourse_marker=is_discourse_marker,
                    ))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern '{text}': {e}")
    
    # Sort by pattern length (longest first) to ensure longer matches take precedence
    patterns.sort(key=lambda p: len(p.text), reverse=True)
    
    return patterns


# Initialize compiled patterns
_compiled_patterns = _compile_patterns()


# =============================================================================
# Sentence Segmentation
# =============================================================================

# Simple sentence boundary pattern for fallback
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _segment_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Segment text into sentences.
    
    Returns:
        List of (start, end, text) tuples for each sentence.
    """
    if not text:
        return []
    
    sentences = []
    last_end = 0
    
    for match in _SENTENCE_BOUNDARY.finditer(text):
        sent_text = text[last_end:match.start() + 1].strip()
        if sent_text:
            sentences.append((last_end, match.start() + 1, sent_text))
        last_end = match.end()
    
    # Last sentence
    if last_end < len(text):
        sent_text = text[last_end:].strip()
        if sent_text:
            sentences.append((last_end, len(text), sent_text))
    
    return sentences if sentences else [(0, len(text), text)]


def _get_sentence_for_position(
    text: str, 
    pos: int, 
    sentences: Optional[List[Tuple[int, int, str]]] = None
) -> Tuple[str, int, bool]:
    """
    Get the sentence containing a character position.
    
    Returns:
        Tuple of (sentence_text, sentence_start, is_at_sentence_start)
    """
    if sentences is None:
        sentences = _segment_sentences(text)
    
    for start, end, sent_text in sentences:
        if start <= pos < end:
            # Check if position is at sentence start (within first few chars)
            relative_pos = pos - start
            # Account for leading whitespace in the sentence
            stripped_start = len(sent_text) - len(sent_text.lstrip())
            is_at_start = relative_pos <= stripped_start + 1
            return (sent_text, start, is_at_start)
    
    # Fallback: return context around position
    context_chars = 100
    start = max(0, pos - context_chars)
    end = min(len(text), pos + context_chars)
    return (text[start:end], start, pos == start)


# =============================================================================
# Composite Pattern Suppression
# =============================================================================

# Words that should be suppressed when part of a composite pattern
# NOTE: "between" is NOT suppressed because we need signal-guided merging to 
# combine "between" + span + "and" + span → Between(span1, span2)
# TEI2GO doesn't detect "between X and Y" as a single span, so we need the signal.
_COMPOSITE_SIGNAL_WORDS = {
    # 'between' - NOT suppressed, needed for signal-guided merge
    'early', 'late', 'mid',  # Modifiers - suppressed because TEI2GO handles "early morning of X"
    'around', 'approximately', 'about', 'roughly', 'circa',  # Approximation - TEI2GO handles "around X"
}

# Time-of-day words that combine with modifiers
_TIME_OF_DAY_WORDS = {
    'morning', 'afternoon', 'evening', 'night', 'dawn', 'noon', 'midnight'
}

# Month names for modifier patterns
_MONTH_NAMES = {
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
}


def _is_part_of_composite(signal_text: str, full_text: str, start: int, end: int) -> bool:
    """
    Check if a signal word is part of a composite temporal pattern.
    
    When a signal is part of a composite (like "between" in "between X and Y"),
    it should be suppressed as a separate signal because the entire expression
    will be parsed as a unified SCATEX expression.
    
    Args:
        signal_text: The matched signal text
        full_text: The full text being analyzed
        start: Start position of the signal in full_text
        end: End position of the signal in full_text
        
    Returns:
        True if the signal is part of a composite pattern (should be suppressed),
        False otherwise (should be emitted as a signal).
    """
    text_lower = signal_text.lower().strip()
    
    # Only check words that can be part of composite patterns
    if text_lower not in _COMPOSITE_SIGNAL_WORDS:
        return False
    
    # Get context after the signal (look ahead up to 100 chars)
    context_after = full_text[end:end + 100].lower() if end < len(full_text) else ""
    
    # === Modifiers: "early", "late", "mid" ===
    if text_lower in ('early', 'late', 'mid'):
        # Check for time-of-day: "early morning", "late evening"
        # Allow optional "in the" between modifier and time-of-day
        tod_pattern = r'^\s*(?:in\s+the\s+)?(' + '|'.join(_TIME_OF_DAY_WORDS) + r')\b'
        if re.match(tod_pattern, context_after):
            return True
        
        # Check for month: "early October", "late January"
        month_pattern = r'^\s*[- ]?(' + '|'.join(_MONTH_NAMES) + r')\b'
        if re.match(month_pattern, context_after):
            return True
        
        # Check for year: "early 2023", "late 2020"
        if re.match(r'^\s*[- ]?\d{4}\b', context_after):
            return True
    
    # === Approximation words: "around", "approximately", "circa", etc. ===
    if text_lower in ('around', 'approximately', 'about', 'roughly', 'circa'):
        # Check if followed by something that looks like a date/time
        # Numbers, month names, or words like "the", "next", "last"
        approx_patterns = [
            r'^\s+\d',  # Followed by a number
            r'^\s+(' + '|'.join(_MONTH_NAMES) + r')\b',  # Month name
            r'^\s+(the|next|last|this)\s+',  # Date phrases
            r'^\s+\d{1,2}:\d{2}',  # Time like "6:30"
            r'^\s+(noon|midnight|morning|evening|afternoon|night|dawn)',  # Time of day
        ]
        for pattern in approx_patterns:
            if re.match(pattern, context_after, re.I):
                return True
    
    return False


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_temporal_signals(text: str) -> List[TemporalSignal]:
    """
    Extract temporal signals from text.
    
    Uses two complementary approaches:
    1. Lexicon matching - catches fixed phrases and idioms
    2. Template matching - catches productive patterns like "N units before/after"
    
    Args:
        text: The input text to analyze.
        
    Returns:
        List of TemporalSignal objects with detected signals.
    """
    if not text or not text.strip():
        return []
    
    signals = []
    seen_ranges: Set[Tuple[int, int]] = set()
    
    # Pre-segment sentences for efficiency
    sentences = _segment_sentences(text)
    
    # Step 1: Template matching first (more specific patterns)
    # This catches "two days before", "the following week", etc.
    template_signals = _extract_template_matches(text, sentences, seen_ranges)
    signals.extend(template_signals)
    
    # Step 2: Lexicon matching (fixed phrases and discourse markers)
    for cp in _compiled_patterns:
        for match in cp.pattern.finditer(text):
            start, end = match.span(1) if match.lastindex else match.span()
            
            # Skip if overlapping with existing match
            overlaps = False
            for seen_start, seen_end in seen_ranges:
                if start < seen_end and end > seen_start:
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            # Get the matched text
            matched_text = match.group(1) if match.lastindex else match.group()
            
            # Skip if this signal is part of a composite pattern
            # (e.g., "between" in "between X and Y" should be suppressed)
            if _is_part_of_composite(matched_text, text, start, end):
                logger.debug(f"Suppressing signal '{matched_text}' - part of composite pattern")
                continue
            
            # Get sentence context
            sentence, sent_start, is_at_sent_start = _get_sentence_for_position(
                text, start, sentences
            )
            
            # For discourse markers, verify they're actually at sentence start
            if cp.is_discourse_marker and not is_at_sent_start:
                continue
            
            signals.append(TemporalSignal(
                text=matched_text,
                start=start,
                end=end,
                relation=cp.relation,
                category=cp.category,
                sentence=sentence,
                sentence_start=sent_start,
                is_anaphoric=cp.is_anaphoric,
                is_sentence_initial=is_at_sent_start and cp.is_discourse_marker,
            ))
            seen_ranges.add((start, end))
    
    # Sort by position
    signals.sort(key=lambda s: s.start)
    
    # Filter out ambiguous signals that are likely not temporal
    signals = _filter_ambiguous_signals(signals)
    
    return signals


def extract_temporal_signals_batch(
    texts: List[str],
    batch_size: int = 100  # Not used, but kept for API consistency
) -> List[List[TemporalSignal]]:
    """
    Extract temporal signals from multiple texts.
    
    Args:
        texts: List of input texts.
        batch_size: Ignored (for API consistency with tei2go).
        
    Returns:
        List of lists of TemporalSignal objects (one list per input text).
    """
    return [extract_temporal_signals(text) for text in texts]


# =============================================================================
# Disambiguation Functions
# =============================================================================

# Ambiguous single-word prepositions that often have non-temporal meanings
AMBIGUOUS_SIGNALS = {"by", "as", "from", "at", "in", "on", "to"}

# Minimum length for signals that don't require disambiguation
MIN_UNAMBIGUOUS_LENGTH = 3  # Signals longer than this are kept without checking

# Words that commonly precede "by" in non-temporal agent constructions
AGENT_VERBS = {
    "killed", "murdered", "attacked", "captured", "taken", "seized",
    "made", "done", "created", "written", "built", "designed", "produced",
    "led", "headed", "commanded", "directed", "organized", "founded",
    "published", "released", "announced", "reported", "discovered",
    "wounded", "injured", "shot", "abducted", "kidnapped", "rescued"
}

# Temporal indicators that follow prepositions (making them temporal)
TEMPORAL_FOLLOWERS = [
    # Time words
    "the time", "then", "now", "that point", "this point", "that time", "this time",
    "morning", "afternoon", "evening", "night", "noon", "midnight", "dawn", "dusk", "sunrise", "sunset",
    # Days
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Months
    "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december",
    # Units
    "day", "week", "month", "year", "hour", "minute", "second",
    "days", "weeks", "months", "years", "hours", "minutes", "seconds",
    # Ordinals
    "the 1st", "the 2nd", "the 3rd", "first", "second", "third",
    # Relative
    "the next", "the previous", "the following", "the preceding",
    "the end", "the start", "the beginning", "the middle",
]

# Non-temporal phrases that start with ambiguous prepositions
NON_TEMPORAL_PHRASES = {
    "as": ["part of", "well as", "such", "a result", "long as", "far as", "if", "though", "much as"],
    "by": ["the", "a ", "an ", "his", "her", "their", "its", "our", "your"],  # Agent markers
    "from": ["the", "a ", "an "],  # Often spatial, but check further
    "at": ["the", "a ", "an ", "his", "her", "their", "least", "most", "all"],
    "in": ["the", "a ", "an ", "his", "her", "their", "order to", "case", "fact", "addition"],
    "on": ["the", "a ", "an ", "his", "her", "their"],
    "to": ["the", "a ", "an ", "his", "her", "their", "be", "have", "do", "make"],
}


def _is_likely_temporal_usage(
    signal_text: str, 
    sentence: str, 
    start_in_sentence: int
) -> bool:
    """
    Determine if an ambiguous preposition is likely being used temporally.
    
    Args:
        signal_text: The matched signal text
        sentence: The containing sentence
        start_in_sentence: Position of signal within the sentence
        
    Returns:
        True if the signal is likely temporal, False if it should be filtered out
    """
    text_lower = signal_text.lower()
    
    # Non-ambiguous signals are always kept
    if text_lower not in AMBIGUOUS_SIGNALS:
        return True
    
    # Get context around the signal
    before_text = sentence[:start_in_sentence].rstrip().lower()
    after_start = start_in_sentence + len(signal_text)
    after_text = sentence[after_start:].lstrip().lower()[:40]  # Look ahead 40 chars
    
    # Check for explicit temporal followers (strong positive signal)
    for indicator in TEMPORAL_FOLLOWERS:
        if after_text.startswith(indicator.lower()):
            return True
    
    # Check for time patterns: digits followed by colon (e.g., "at 6:30")
    if re.match(r'\d{1,2}:\d{2}', after_text):
        return True
    
    # Check for year patterns (e.g., "in 2023")
    if re.match(r'\d{4}\b', after_text):
        return True
    
    # Check for digit + unit patterns (e.g., "in 5 minutes")
    if re.match(r'\d+\s+(day|week|month|year|hour|minute|second)s?\b', after_text):
        return True
    
    # === Specific disambiguation rules per preposition ===
    
    if text_lower == "by":
        # "by" is often non-temporal when preceded by passive voice verbs
        # E.g., "killed by", "made by", "attacked by"
        for verb in AGENT_VERBS:
            if before_text.endswith(verb) or before_text.endswith(verb + "ed"):
                return False
        # "by" followed by a determiner + noun is usually an agent, not temporal
        # E.g., "by Israeli forces", "by the attackers"
        if re.match(r'(the|a|an|his|her|their|its)\s+\w+', after_text):
            if not any(after_text.startswith(t) for t in TEMPORAL_FOLLOWERS):
                return False
    
    elif text_lower == "as":
        # "as" is non-temporal in these constructions
        for phrase in NON_TEMPORAL_PHRASES.get("as", []):
            if after_text.startswith(phrase.lower()):
                return False
    
    elif text_lower == "from":
        # "from" is spatial when followed by a location or entity
        # E.g., "from Gaza", "from Israel", "from the north"
        # But temporal when "from the start", "from day one", etc.
        for phrase in ["the start", "the beginning", "the outset", "day one", "the get-go"]:
            if after_text.startswith(phrase.lower()):
                return True
        # If followed by a capitalized word (likely a location/entity), probably spatial
        after_stripped = sentence[after_start:].lstrip()
        if after_stripped and after_stripped[0].isupper():
            # Check if it's not a temporal word that happens to be capitalized
            after_word = re.match(r'[A-Z][a-z]+', after_stripped)
            if after_word:
                word = after_word.group().lower()
                if word not in ["monday", "tuesday", "wednesday", "thursday", "friday", 
                               "saturday", "sunday", "january", "february", "march", 
                               "april", "may", "june", "july", "august", "september",
                               "october", "november", "december"]:
                    return False
    
    elif text_lower == "at":
        # "at" is temporal before times, non-temporal before places
        # "at 6:30" (temporal) vs "at the festival" (spatial)
        if re.match(r'\d', after_text):
            return True  # Likely time
        if re.match(r'(the|a|an)\s+', after_text):
            # "at the time", "at the moment" are temporal
            if any(after_text.startswith(t) for t in ["the time", "the moment", "that time", 
                                                       "this time", "that point", "this point",
                                                       "the same time", "the start", "the end"]):
                return True
            return False  # Likely spatial
    
    elif text_lower in ("in", "on", "to"):
        # These need explicit temporal indicators to be considered temporal
        # Otherwise they're usually spatial/other
        # We already checked for temporal followers above
        # If we got here, there's no clear temporal indicator
        return False
    
    # Default: keep signals that aren't clearly non-temporal
    # This is conservative - we'd rather have some false positives than miss real signals
    return True


def _filter_ambiguous_signals(signals: List[TemporalSignal]) -> List[TemporalSignal]:
    """
    Filter out signals that are likely not temporal based on context.
    
    This removes common false positives like:
    - "by" in "killed by terrorists" (agent, not temporal)
    - "as" in "as part of" (manner, not temporal)
    - "from" in "from Gaza" (spatial, not temporal)
    
    Args:
        signals: List of detected temporal signals
        
    Returns:
        Filtered list with ambiguous non-temporal signals removed
    """
    filtered = []
    for signal in signals:
        # Calculate position within sentence
        start_in_sentence = signal.start - signal.sentence_start
        
        if _is_likely_temporal_usage(signal.text, signal.sentence, start_in_sentence):
            filtered.append(signal)
        else:
            logger.debug(
                f"Filtered ambiguous signal '{signal.text}' in: "
                f"'{signal.sentence[:60]}...'"
            )
    
    return filtered


# =============================================================================
# Filtering Functions
# =============================================================================

def filter_by_relation(
    signals: List[TemporalSignal],
    relations: List[TemporalRelation]
) -> List[TemporalSignal]:
    """Filter signals by relation type."""
    return [s for s in signals if s.relation in relations]


def filter_by_category(
    signals: List[TemporalSignal],
    categories: List[SignalCategory]
) -> List[TemporalSignal]:
    """Filter signals by grammatical category."""
    return [s for s in signals if s.category in categories]


def filter_anaphoric(
    signals: List[TemporalSignal],
    anaphoric_only: bool = True
) -> List[TemporalSignal]:
    """Filter signals by whether they require anaphoric resolution."""
    return [s for s in signals if s.is_anaphoric == anaphoric_only]


def filter_discourse_markers(
    signals: List[TemporalSignal]
) -> List[TemporalSignal]:
    """Get only sentence-initial discourse markers."""
    return [s for s in signals if s.is_sentence_initial]


# =============================================================================
# Utility Functions
# =============================================================================

def get_signal_lexicon() -> Dict[str, List[str]]:
    """
    Get the full signal lexicon as a simple dictionary.
    
    Returns:
        Dict mapping relation names to lists of signal texts.
    """
    result = {}
    for relation, categories in SIGNAL_LEXICON.items():
        signals = []
        for entries in categories.values():
            signals.extend([text for text, _ in entries])
        result[relation.value] = signals
    return result


def get_relation_for_signal(signal_text: str) -> Optional[TemporalRelation]:
    """
    Look up the temporal relation for a given signal text.
    
    Args:
        signal_text: The signal text to look up.
        
    Returns:
        The TemporalRelation, or None if not found.
    """
    signal_lower = signal_text.lower()
    
    for relation, categories in SIGNAL_LEXICON.items():
        for entries in categories.values():
            for text, _ in entries:
                if text.lower() == signal_lower:
                    return relation
    
    return None


def count_signals_by_relation(signals: List[TemporalSignal]) -> Dict[str, int]:
    """Count signals grouped by relation type."""
    counts: Dict[str, int] = {}
    for signal in signals:
        key = signal.relation.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_signals_by_category(signals: List[TemporalSignal]) -> Dict[str, int]:
    """Count signals grouped by grammatical category."""
    counts: Dict[str, int] = {}
    for signal in signals:
        key = signal.category.value
        counts[key] = counts.get(key, 0) + 1
    return counts

