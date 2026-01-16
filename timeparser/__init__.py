__version__ = "1.2.2"

from .conf import apply_settings
from .date import DateDataParser, ScatexData

# Re-export SCATEX types for convenience
from .scatex import (
    TemporalExpression,
    # Core intervals
    Year, Month, Day, Hour, Minute, Second, Instant, Interval,
    # Repeating patterns
    Repeating, DayOfWeek, MonthOfYear, TimeOfDay, RepeatingIntersection,
    # Relative operators
    This, Last, Next, Shift, Period, Before, After, Between,
    # Composite patterns
    ModifiedInterval, Approximate,
    # Set operations
    Union, Intersection,
    # Convenience expressions
    Now, Today, Yesterday, Tomorrow, Decade, Century, Quarter, Unknown,
    # Enums
    Unit, DayOfWeekType, MonthOfYearType, TimeOfDayType, Direction,
    # Evaluation helper
    evaluate_scatex_code, format_interval,
)

# =============================================================================
# Detection Module Exports
# =============================================================================

# tei2go exports (spaCy-based NER span detection)
from .detection.tei2go import (
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

# temporal_signals exports (regex-based signal/discourse marker detection)
from .detection.temporal_signals import (
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

# composite_patterns exports (multi-part temporal expression detection)
from .composite_patterns import (
    # NEW: Signal-guided span merging (preferred approach)
    merge_signals_with_spans,
    MergeResult,
    # Utilities
    filter_spans_covered_by_composites,
    CompositeSpan,
    try_parse_composite,
    # DEPRECATED: extract_composite_spans (use merge_signals_with_spans instead)
    extract_composite_spans,
)

_default_parser = DateDataParser()


@apply_settings
def parse(
    date_string,
    date_formats=None,
    languages=None,
    locales=None,
    region=None,
    settings=None,
    detect_languages_function=None,
):
    """Parse date and time from given date string and return a SCATEX expression.

    SCATEX (SCATE eXecutable) is a compositional representation of temporal
    expressions that can be evaluated with an anchor datetime to get concrete
    (start, end) intervals.

    :param date_string:
        A string representing date and/or time in a recognizably valid format.
    :type date_string: str

    :param date_formats:
        A list of format strings using directives as given
        `here <https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior>`_.
        The parser applies formats one by one, taking into account the detected languages/locales.
    :type date_formats: list

    :param languages:
        A list of language codes, e.g. ['en', 'es', 'zh-Hant'].
        If locales are not given, languages and region are used to construct locales for translation.
    :type languages: list

    :param locales:
        A list of locale codes, e.g. ['fr-PF', 'qu-EC', 'af-NA'].
        The parser uses only these locales to translate date string.
    :type locales: list

    :param region:
        A region code, e.g. 'IN', '001', 'NE'.
        If locales are not given, languages and region are used to construct locales for translation.
    :type region: str

    :param settings:
        Configure customized behavior using settings defined in :mod:`timeparser.conf.Settings`.
    :type settings: dict

    :param detect_languages_function:
        A function for language detection that takes as input a string (the `date_string`) and
        a `confidence_threshold`, and returns a list of detected language codes.
        Note: this function is only used if ``languages`` and ``locales`` are not provided.
    :type detect_languages_function: function

    :return: Returns a SCATEX TemporalExpression if parsing is successful, else returns None.
        The expression can be evaluated with .evaluate(anchor_datetime) to get (start, end) datetimes.
    :rtype: TemporalExpression or None
    
    :raises:
        ``ValueError``: Unknown Language, ``TypeError``: Languages argument must be a list,
        ``SettingValidationError``: A provided setting is not valid.

    Example usage::

        >>> import timeparser
        >>> from datetime import datetime
        
        # Parse returns a SCATEX expression
        >>> expr = timeparser.parse("October 7, 2023")
        >>> expr
        Day(day=7, month=10, year=2023)
        
        # Evaluate with an anchor datetime
        >>> start, end = expr.evaluate(datetime.now())
        >>> start
        datetime.datetime(2023, 10, 7, 0, 0, 0)
        
        # Relative dates work too
        >>> expr = timeparser.parse("3 days ago")
        >>> expr
        Shift(interval=Today(), period=Period(unit=DAY, value=3), direction=BEFORE)
        
        # Partial dates preserve missing components
        >>> expr = timeparser.parse("October 7")
        >>> expr
        Day(day=7, month=10)  # year is None
    """
    parser = _default_parser

    if (
        languages
        or locales
        or region
        or detect_languages_function
        or not settings._default
    ):
        parser = DateDataParser(
            languages=languages,
            locales=locales,
            region=region,
            settings=settings,
            detect_languages_function=detect_languages_function,
        )

    data = parser.get_scatex_data(date_string, date_formats)

    if data:
        return data["scatex_expr"]
    return None
