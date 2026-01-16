import collections
from collections.abc import Set
from datetime import datetime, timedelta
from typing import Optional, Union

import regex as re
from dateutil.relativedelta import relativedelta
from tzlocal import get_localzone

from timeparser.conf import apply_settings, check_settings
from timeparser.custom_language_detection.language_mapping import map_languages
from timeparser.date_parser import date_parser
from timeparser.freshness_date_parser import freshness_date_parser
from timeparser.languages.loader import LocaleDataLoader
from timeparser.parser import _parse_absolute, _parse_nospaces, _parse_absolute_scatex
from timeparser.timezone_parser import pop_tz_offset_from_string
from timeparser.utils import (
    apply_timezone_from_settings,
    get_timezone_from_tz_string,
    set_correct_day_from_settings,
    set_correct_month_from_settings,
)
from timeparser.utils.strptime import strptime as patched_strptime

# Import SCATEX types
from timeparser.scatex import (
    TemporalExpression,
    Year, Month, Day, Hour, Minute, Second, Instant, Interval,
    Repeating, DayOfWeek, MonthOfYear, TimeOfDay, RepeatingIntersection,
    This, Last, Next, Shift, Period, Before, After, Between,
    Union as ScatexUnion, Intersection,
    Now, Today, Yesterday, Tomorrow, Decade, Century, Quarter, Unknown,
    Unit, DayOfWeekType, MonthOfYearType, TimeOfDayType, Direction,
)

APOSTROPHE_LOOK_ALIKE_CHARS = [
    "\N{RIGHT SINGLE QUOTATION MARK}",  # '\u2019'
    "\N{MODIFIER LETTER APOSTROPHE}",  # '\u02bc'
    "\N{MODIFIER LETTER TURNED COMMA}",  # '\u02bb'
    "\N{ARMENIAN APOSTROPHE}",  # '\u055a'
    "\N{LATIN SMALL LETTER SALTILLO}",  # '\ua78c'
    "\N{PRIME}",  # '\u2032'
    "\N{REVERSED PRIME}",  # '\u2035'
    "\N{MODIFIER LETTER PRIME}",  # '\u02b9'
    "\N{FULLWIDTH APOSTROPHE}",  # '\uff07'
]

RE_NBSP = re.compile("\xa0", flags=re.UNICODE)
RE_SPACES = re.compile(r"\s+")
RE_TRIM_SPACES = re.compile(r"^\s+(\S.*?)\s+$")
RE_TRIM_COLONS = re.compile(r"(\S.*?):*$")

RE_SANITIZE_SKIP = re.compile(
    r"\t|\n|\r|\u00bb|,\s\u0432\b|\u200e|\xb7|\u200f|\u064e|\u064f", flags=re.M
)
RE_SANITIZE_RUSSIAN = re.compile(r"([\W\d])\u0433\.", flags=re.I | re.U)
RE_SANITIZE_CROATIAN = re.compile(
    r"(\d+)\.\s?(\d+)\.\s?(\d+)\.( u)?", flags=re.I | re.U
)
RE_SANITIZE_PERIOD = re.compile(r"(?<=[^0-9\s])\.", flags=re.U)
RE_SANITIZE_ON = re.compile(r"^.*?on:\s+(.*)")
RE_SANITIZE_APOSTROPHE = re.compile("|".join(APOSTROPHE_LOOK_ALIKE_CHARS))

RE_SEARCH_TIMESTAMP = re.compile(r"^(\d{10})(\d{3})?(\d{3})?(?![^.])")
RE_SEARCH_NEGATIVE_TIMESTAMP = re.compile(r"^([-]\d{10})(\d{3})?(\d{3})?(?![^.])")


def sanitize_spaces(date_string):
    date_string = RE_NBSP.sub(" ", date_string)
    date_string = RE_SPACES.sub(" ", date_string)
    date_string = RE_TRIM_SPACES.sub(r"\1", date_string)
    return date_string


def date_range(begin, end, **kwargs):
    dateutil_error_prone_args = [
        "year",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
    ]
    for arg in dateutil_error_prone_args:
        if arg in kwargs:
            raise ValueError("Invalid argument: %s" % arg)

    step = relativedelta(**kwargs) if kwargs else relativedelta(days=1)

    date = begin
    while date < end:
        yield date
        date += step

    # handles edge-case when iterating months and last interval is < 30 days
    if kwargs.get("months", 0) > 0 and (date.year, date.month) == (end.year, end.month):
        yield end


def get_intersecting_periods(low, high, period="day"):
    if period not in [
        "year",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
    ]:
        raise ValueError("Invalid period: {}".format(period))

    if high <= low:
        return

    step = relativedelta(**{period + "s": 1})

    current_period_start = low
    if isinstance(current_period_start, datetime):
        reset_arguments = {}
        for test_period in ["microsecond", "second", "minute", "hour"]:
            if test_period == period:
                break
            else:
                reset_arguments[test_period] = 0
        current_period_start = current_period_start.replace(**reset_arguments)

    if period == "week":
        current_period_start = current_period_start - timedelta(
            days=current_period_start.weekday()
        )
    elif period == "month":
        current_period_start = current_period_start.replace(day=1)
    elif period == "year":
        current_period_start = current_period_start.replace(month=1, day=1)

    while current_period_start < high:
        yield current_period_start
        current_period_start += step


def sanitize_date(date_string):
    date_string = RE_SANITIZE_SKIP.sub(" ", date_string)
    date_string = RE_SANITIZE_RUSSIAN.sub(
        r"\1 ", date_string
    )  # remove 'г.' (Russian for year) but not in words
    date_string = RE_SANITIZE_CROATIAN.sub(
        r"\1.\2.\3 ", date_string
    )  # extra '.' and 'u' interferes with parsing relative fractional dates
    date_string = sanitize_spaces(date_string)
    date_string = RE_SANITIZE_PERIOD.sub("", date_string)
    date_string = RE_SANITIZE_ON.sub(r"\1", date_string)
    date_string = RE_TRIM_COLONS.sub(r"\1", date_string)
    date_string = RE_SANITIZE_APOSTROPHE.sub("'", date_string)
    date_string = date_string.strip()
    return date_string


def get_date_from_timestamp(date_string, settings, negative=False):
    if negative:
        match = RE_SEARCH_NEGATIVE_TIMESTAMP.search(date_string)
    else:
        match = RE_SEARCH_TIMESTAMP.search(date_string)

    if match:
        if (
            settings is None
            or settings.TIMEZONE is None
            or "local" in settings.TIMEZONE.lower()
        ):
            # If the timezone in settings is unset, or it's 'local', use the
            # local timezone
            timezone = get_localzone()
        else:
            # Otherwise, use the timezone given in settings
            timezone = get_timezone_from_tz_string(settings.TIMEZONE)

        seconds = int(match.group(1))
        millis = int(match.group(2) or 0)
        micros = int(match.group(3) or 0)
        date_obj = datetime.fromtimestamp(seconds, timezone).replace(
            microsecond=millis * 1000 + micros, tzinfo=None
        )
        date_obj = apply_timezone_from_settings(date_obj, settings)
        return date_obj


def parse_with_formats(date_string, date_formats, settings):
    """Parse with formats and return a dictionary with 'period' and 'obj_date'.

    :returns: :class:`datetime.datetime`, dict or None

    """
    period = "day"
    for date_format in date_formats:
        try:
            date_obj = patched_strptime(date_string, date_format)
        except ValueError:
            continue
        else:
            missing_month = not any(m in date_format for m in ["%m", "%b", "%B"])
            missing_day = "%d" not in date_format
            if missing_month and missing_day:
                period = "year"
                date_obj = set_correct_month_from_settings(date_obj, settings)
                date_obj = set_correct_day_from_settings(date_obj, settings)

            elif missing_month:
                period = "year"
                date_obj = set_correct_month_from_settings(date_obj, settings)

            elif missing_day:
                period = "month"
                date_obj = set_correct_day_from_settings(date_obj, settings)

            if not ("%y" in date_format or "%Y" in date_format):
                today = datetime.today()
                date_obj = date_obj.replace(year=today.year)

            date_obj = apply_timezone_from_settings(date_obj, settings)

            return DateData(date_obj=date_obj, period=period)
    else:
        return DateData(date_obj=None, period=period)


class _DateLocaleParser:
    def __init__(self, locale, date_string, date_formats, settings=None):
        self._settings = settings
        if not (date_formats is None or isinstance(date_formats, (list, tuple, Set))):
            raise TypeError("Date formats should be list, tuple or set of strings")

        self.locale = locale
        self.date_string = date_string
        self.date_formats = date_formats
        self._translated_date = None
        self._translated_date_with_formatting = None
        self._parsers = {
            "timestamp": self._try_timestamp,
            "negative-timestamp": self._try_negative_timestamp,
            "relative-time": self._try_freshness_parser,
            "custom-formats": self._try_given_formats,
            "absolute-time": self._try_absolute_parser,
            "no-spaces-time": self._try_nospaces_parser,
        }

    @classmethod
    def parse(cls, locale, date_string, date_formats=None, settings=None):
        instance = cls(locale, date_string, date_formats, settings)
        return instance._parse()

    def _parse(self):
        for parser_name in self._settings.PARSERS:
            date_data = self._parsers[parser_name]()
            if self._is_valid_date_data(date_data):
                return date_data
        else:
            return None

    def _try_timestamp_parser(self, negative=False):
        return DateData(
            date_obj=get_date_from_timestamp(
                self.date_string, self._settings, negative=negative
            ),
            period="time" if self._settings.RETURN_TIME_AS_PERIOD else "day",
        )

    def _try_timestamp(self):
        return self._try_timestamp_parser()

    def _try_negative_timestamp(self):
        return self._try_timestamp_parser(negative=True)

    def _try_freshness_parser(self):
        try:
            return freshness_date_parser.get_date_data(
                self._get_translated_date(), self._settings
            )
        except (OverflowError, ValueError):
            return None

    def _try_absolute_parser(self):
        return self._try_parser(parse_method=_parse_absolute)

    def _try_nospaces_parser(self):
        return self._try_parser(parse_method=_parse_nospaces)

    def _try_parser(self, parse_method):
        _order = self._settings.DATE_ORDER
        try:
            if self._settings.PREFER_LOCALE_DATE_ORDER:
                if "DATE_ORDER" not in self._settings._mod_settings:
                    self._settings.DATE_ORDER = self.locale.info.get(
                        "date_order", _order
                    )
            date_obj, period = date_parser.parse(
                self._get_translated_date(),
                parse_method=parse_method,
                settings=self._settings,
            )
            self._settings.DATE_ORDER = _order
            return DateData(
                date_obj=date_obj,
                period=period,
            )
        except ValueError:
            self._settings.DATE_ORDER = _order
            return None

    def _try_given_formats(self):
        if not self.date_formats:
            return

        return parse_with_formats(
            self._get_translated_date_with_formatting(),
            self.date_formats,
            settings=self._settings,
        )

    def _get_translated_date(self):
        if self._translated_date is None:
            self._translated_date = self.locale.translate(
                self.date_string, keep_formatting=False, settings=self._settings
            )
        return self._translated_date

    def _get_translated_date_with_formatting(self):
        if self._translated_date_with_formatting is None:
            self._translated_date_with_formatting = self.locale.translate(
                self.date_string, keep_formatting=True, settings=self._settings
            )
        return self._translated_date_with_formatting

    def _is_valid_date_data(self, date_data):
        if not isinstance(date_data, DateData):
            return False
        if not date_data["date_obj"] or not date_data["period"]:
            return False
        if date_data["date_obj"] and not isinstance(date_data["date_obj"], datetime):
            return False
        if date_data["period"] not in ("time", "day", "week", "month", "year"):
            return False
        return True


class _ScatexLocaleParser:
    """Parser that outputs SCATEX expressions instead of datetime objects."""
    
    def __init__(self, locale, date_string, date_formats, settings=None):
        self._settings = settings
        if not (date_formats is None or isinstance(date_formats, (list, tuple, Set))):
            raise TypeError("Date formats should be list, tuple or set of strings")

        self.locale = locale
        self.date_string = date_string
        self.date_formats = date_formats
        self._translated_date = None
        self._translated_date_with_formatting = None
        self._parsers = {
            "timestamp": self._try_timestamp,
            "negative-timestamp": self._try_negative_timestamp,
            "relative-time": self._try_freshness_parser,
            "custom-formats": self._try_given_formats,
            "absolute-time": self._try_absolute_parser,
            "no-spaces-time": self._try_nospaces_parser,
        }

    @classmethod
    def parse(cls, locale, date_string, date_formats=None, settings=None):
        instance = cls(locale, date_string, date_formats, settings)
        return instance._parse()

    def _parse(self):
        # =====================================================================
        # Try composite patterns FIRST (highest priority)
        # This handles expressions like "between X and Y", "early morning of X"
        # =====================================================================
        from timeparser.composite_patterns import composite_parser
        composite_result = composite_parser.try_parse(self.date_string, self._settings)
        if composite_result:
            expr, period = composite_result
            return ScatexData(scatex_expr=expr, period=period)
        
        # Continue with standard parser chain
        for parser_name in self._settings.PARSERS:
            scatex_data = self._parsers[parser_name]()
            if self._is_valid_scatex_data(scatex_data):
                return scatex_data
        return None

    def _try_timestamp_parser(self, negative=False):
        scatex_expr = get_scatex_from_timestamp(
            self.date_string, self._settings, negative=negative
        )
        if scatex_expr:
            return ScatexData(
                scatex_expr=scatex_expr,
                period="time" if self._settings.RETURN_TIME_AS_PERIOD else "day",
            )
        return ScatexData(scatex_expr=None, period="day")

    def _try_timestamp(self):
        return self._try_timestamp_parser()

    def _try_negative_timestamp(self):
        return self._try_timestamp_parser(negative=True)

    def _try_freshness_parser(self):
        try:
            from timeparser.freshness_date_parser import freshness_date_parser
            return freshness_date_parser.get_scatex_data(
                self._get_translated_date(), self._settings
            )
        except (OverflowError, ValueError):
            return None

    def _try_absolute_parser(self):
        return self._try_parser(parse_method=_parse_absolute_scatex)

    def _try_nospaces_parser(self):
        # For no-spaces parsing, we'll fall back to datetime and convert
        _order = self._settings.DATE_ORDER
        try:
            if self._settings.PREFER_LOCALE_DATE_ORDER:
                if "DATE_ORDER" not in self._settings._mod_settings:
                    self._settings.DATE_ORDER = self.locale.info.get(
                        "date_order", _order
                    )
            from timeparser.parser import _no_spaces_parser
            date_obj, period = _no_spaces_parser.parse(
                self._get_translated_date(), self._settings
            )
            self._settings.DATE_ORDER = _order
            
            # Convert datetime to SCATEX
            if date_obj:
                scatex_expr = build_scatex_from_components(
                    year=date_obj.year,
                    month=date_obj.month,
                    day=date_obj.day,
                    hour=date_obj.hour if date_obj.hour != 0 or date_obj.minute != 0 or date_obj.second != 0 else None,
                    minute=date_obj.minute if date_obj.minute != 0 or date_obj.second != 0 else None,
                    second=date_obj.second if date_obj.second != 0 else None,
                )
                return ScatexData(scatex_expr=scatex_expr, period=period)
            return ScatexData(scatex_expr=None, period=period)
        except ValueError:
            self._settings.DATE_ORDER = _order
            return None

    def _try_parser(self, parse_method):
        _order = self._settings.DATE_ORDER
        try:
            if self._settings.PREFER_LOCALE_DATE_ORDER:
                if "DATE_ORDER" not in self._settings._mod_settings:
                    self._settings.DATE_ORDER = self.locale.info.get(
                        "date_order", _order
                    )
            scatex_expr, period = parse_method(
                self._get_translated_date(),
                settings=self._settings,
            )
            self._settings.DATE_ORDER = _order
            return ScatexData(
                scatex_expr=scatex_expr,
                period=period,
            )
        except ValueError:
            self._settings.DATE_ORDER = _order
            return None

    def _try_given_formats(self):
        if not self.date_formats:
            return None

        return parse_with_formats_scatex(
            self._get_translated_date_with_formatting(),
            self.date_formats,
            settings=self._settings,
        )

    def _get_translated_date(self):
        if self._translated_date is None:
            self._translated_date = self.locale.translate(
                self.date_string, keep_formatting=False, settings=self._settings
            )
        return self._translated_date

    def _get_translated_date_with_formatting(self):
        if self._translated_date_with_formatting is None:
            self._translated_date_with_formatting = self.locale.translate(
                self.date_string, keep_formatting=True, settings=self._settings
            )
        return self._translated_date_with_formatting

    def _is_valid_scatex_data(self, scatex_data):
        if not isinstance(scatex_data, ScatexData):
            return False
        if scatex_data["scatex_expr"] is None or not scatex_data["period"]:
            return False
        if not isinstance(scatex_data["scatex_expr"], TemporalExpression):
            return False
        if scatex_data["period"] not in ("time", "second", "minute", "hour", "day", "week", "month", "year", "decade"):
            return False
        return True


class DateData:
    """
    Class that represents the parsed data with useful information.
    It can be accessed with square brackets like a dict object.
    """

    def __init__(self, *, date_obj=None, period=None, locale=None):
        self.date_obj = date_obj
        self.period = period
        self.locale = locale

    def __getitem__(self, k):
        if not hasattr(self, k):
            raise KeyError(k)
        return getattr(self, k)

    def __setitem__(self, k, v):
        if not hasattr(self, k):
            raise KeyError(k)
        setattr(self, k, v)

    def __repr__(self):
        properties_text = ", ".join(
            "{}={}".format(prop, val.__repr__()) for prop, val in self.__dict__.items()
        )

        return "{}({})".format(self.__class__.__name__, properties_text)


class ScatexData:
    """
    Class that represents parsed SCATEX temporal expression data.
    It can be accessed with square brackets like a dict object.
    
    The scatex_expr is a SCATEX TemporalExpression that can be evaluated
    with an anchor datetime to get concrete (start, end) intervals.
    """

    def __init__(self, *, scatex_expr: Optional[TemporalExpression] = None, 
                 period: Optional[str] = None, locale: Optional[str] = None):
        self.scatex_expr = scatex_expr
        self.period = period
        self.locale = locale

    def __getitem__(self, k):
        if not hasattr(self, k):
            raise KeyError(k)
        return getattr(self, k)

    def __setitem__(self, k, v):
        if not hasattr(self, k):
            raise KeyError(k)
        setattr(self, k, v)

    def __repr__(self):
        properties_text = ", ".join(
            "{}={}".format(prop, val.__repr__()) for prop, val in self.__dict__.items()
        )
        return "{}({})".format(self.__class__.__name__, properties_text)
    
    def evaluate(self, anchor: datetime):
        """
        Evaluate the SCATEX expression with the given anchor datetime.
        
        :param anchor: Reference datetime for resolving relative expressions
        :return: Tuple of (start, end) datetimes
        """
        if self.scatex_expr is None:
            return (None, None)
        return self.scatex_expr.evaluate(anchor)


def build_scatex_from_components(
    year: Optional[int] = None,
    month: Optional[int] = None, 
    day: Optional[int] = None,
    hour: Optional[int] = None,
    minute: Optional[int] = None,
    second: Optional[int] = None,
    weekday: Optional[str] = None,
    time_of_day: Optional[str] = None,
) -> TemporalExpression:
    """
    Build a SCATEX expression from parsed date/time components.
    
    Components can be None for partial dates. The most specific
    available component determines the expression type.
    
    :param year: Year value (e.g., 2023)
    :param month: Month value (1-12)
    :param day: Day value (1-31)
    :param hour: Hour value (0-23)
    :param minute: Minute value (0-59)
    :param second: Second value (0-59)
    :param weekday: Weekday name (e.g., "monday", "friday")
    :param time_of_day: Time of day name (e.g., "morning", "evening")
    :return: SCATEX TemporalExpression
    """
    # Handle weekday references
    if weekday and not (year or month or day):
        weekday_map = {
            'monday': DayOfWeekType.MONDAY,
            'tuesday': DayOfWeekType.TUESDAY,
            'wednesday': DayOfWeekType.WEDNESDAY,
            'thursday': DayOfWeekType.THURSDAY,
            'friday': DayOfWeekType.FRIDAY,
            'saturday': DayOfWeekType.SATURDAY,
            'sunday': DayOfWeekType.SUNDAY,
        }
        weekday_lower = weekday.lower()
        if weekday_lower in weekday_map:
            day_of_week_expr = DayOfWeek(type=weekday_map[weekday_lower])
            
            # If we have time-of-day, intersect with it
            if time_of_day:
                tod_map = {
                    'morning': TimeOfDayType.MORNING,
                    'afternoon': TimeOfDayType.AFTERNOON,
                    'evening': TimeOfDayType.EVENING,
                    'night': TimeOfDayType.NIGHT,
                    'noon': TimeOfDayType.NOON,
                    'midnight': TimeOfDayType.MIDNIGHT,
                    'dawn': TimeOfDayType.DAWN,
                }
                tod_lower = time_of_day.lower()
                if tod_lower in tod_map:
                    return Intersection(intervals=[
                        day_of_week_expr,
                        TimeOfDay(type=tod_map[tod_lower])
                    ])
            return day_of_week_expr
    
    # Build based on most specific component available
    if second is not None:
        return Second(
            second=second,
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            year=year
        )
    elif minute is not None:
        return Minute(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            year=year
        )
    elif hour is not None:
        return Hour(
            hour=hour,
            day=day,
            month=month,
            year=year
        )
    elif day is not None:
        return Day(
            day=day,
            month=month,
            year=year
        )
    elif month is not None:
        return Month(
            month=month,
            year=year
        )
    elif year is not None:
        return Year(digits=year)
    else:
        return Unknown(reason="No date components found")


def get_scatex_from_timestamp(date_string: str, settings, negative: bool = False) -> Optional[TemporalExpression]:
    """
    Parse a timestamp and return a SCATEX Instant expression.
    
    :param date_string: String containing a Unix timestamp
    :param settings: Parser settings
    :param negative: Whether to look for negative timestamps
    :return: Instant expression or None if not a timestamp
    """
    if negative:
        match = RE_SEARCH_NEGATIVE_TIMESTAMP.search(date_string)
    else:
        match = RE_SEARCH_TIMESTAMP.search(date_string)

    if match:
        if (
            settings is None
            or settings.TIMEZONE is None
            or "local" in settings.TIMEZONE.lower()
        ):
            timezone = get_localzone()
        else:
            timezone = get_timezone_from_tz_string(settings.TIMEZONE)

        seconds = int(match.group(1))
        millis = int(match.group(2) or 0)
        micros = int(match.group(3) or 0)
        date_obj = datetime.fromtimestamp(seconds, timezone).replace(
            microsecond=millis * 1000 + micros, tzinfo=None
        )
        date_obj = apply_timezone_from_settings(date_obj, settings)
        
        # Return as an Instant with full datetime info
        return Second(
            second=date_obj.second,
            minute=date_obj.minute,
            hour=date_obj.hour,
            day=date_obj.day,
            month=date_obj.month,
            year=date_obj.year
        )
    return None


def parse_with_formats_scatex(date_string: str, date_formats: list, settings) -> ScatexData:
    """
    Parse with formats and return a ScatexData with SCATEX expression.
    
    :param date_string: The date string to parse
    :param date_formats: List of strptime format strings
    :param settings: Parser settings
    :return: ScatexData with scatex_expr
    """
    period = "day"
    for date_format in date_formats:
        try:
            date_obj = patched_strptime(date_string, date_format)
        except ValueError:
            continue
        else:
            # Determine which components are present in the format
            has_year = "%y" in date_format or "%Y" in date_format
            has_month = any(m in date_format for m in ["%m", "%b", "%B"])
            has_day = "%d" in date_format
            has_hour = "%H" in date_format or "%I" in date_format
            has_minute = "%M" in date_format
            has_second = "%S" in date_format
            
            # Build SCATEX with only the components that were actually parsed
            year = date_obj.year if has_year else None
            month = date_obj.month if has_month else None
            day = date_obj.day if has_day else None
            hour = date_obj.hour if has_hour else None
            minute = date_obj.minute if has_minute else None
            second = date_obj.second if has_second else None
            
            # Determine period
            if not has_month and not has_day:
                period = "year"
            elif not has_day:
                period = "month"
            elif has_hour or has_minute or has_second:
                period = "time" if settings.RETURN_TIME_AS_PERIOD else "day"
            else:
                period = "day"
            
            scatex_expr = build_scatex_from_components(
                year=year, month=month, day=day,
                hour=hour, minute=minute, second=second
            )
            
            return ScatexData(scatex_expr=scatex_expr, period=period)
    
    return ScatexData(scatex_expr=None, period=period)


class DateDataParser:
    """
    Class which handles language detection, translation and subsequent generic parsing of
    string representing date and/or time.

    :param languages:
        A list of language codes, e.g. ['en', 'es', 'zh-Hant'].
        If locales are not given, languages and region are
        used to construct locales for translation.
    :type languages: list

    :param locales:
        A list of locale codes, e.g. ['fr-PF', 'qu-EC', 'af-NA'].
        The parser uses only these locales to translate date string.
    :type locales: list

    :param region:
        A region code, e.g. 'IN', '001', 'NE'.
        If locales are not given, languages and region are
        used to construct locales for translation.
    :type region: str

    :param try_previous_locales:
        If True, locales previously used to translate date are tried first.
    :type try_previous_locales: bool

    :param use_given_order:
        If True, locales are tried for translation of date string
        in the order in which they are given.
    :type use_given_order: bool

    :param settings:
        Configure customized behavior using settings defined in :mod:`timeparser.conf.Settings`.
    :type settings: dict

    :param detect_languages_function:
        A function for language detection that takes as input a `text` and a `confidence_threshold`,
        and returns a list of detected language codes.
        Note: this function is only used if ``languages`` and ``locales`` are not provided.
    :type detect_languages_function: function

    :return: A parser instance

    :raises:
         ``ValueError``: Unknown Language, ``TypeError``: Languages argument must be a list,
         ``SettingValidationError``: A provided setting is not valid.
    """

    locale_loader = None

    @apply_settings
    def __init__(
        self,
        languages=None,
        locales=None,
        region=None,
        try_previous_locales=False,
        use_given_order=False,
        settings=None,
        detect_languages_function=None,
    ):
        if languages is not None and not isinstance(languages, (list, tuple, Set)):
            raise TypeError(
                "languages argument must be a list (%r given)" % type(languages)
            )

        if locales is not None and not isinstance(locales, (list, tuple, Set)):
            raise TypeError(
                "locales argument must be a list (%r given)" % type(locales)
            )

        if region is not None and not isinstance(region, str):
            raise TypeError("region argument must be str (%r given)" % type(region))

        if not isinstance(try_previous_locales, bool):
            raise TypeError(
                "try_previous_locales argument must be a boolean (%r given)"
                % type(try_previous_locales)
            )

        if not isinstance(use_given_order, bool):
            raise TypeError(
                "use_given_order argument must be a boolean (%r given)"
                % type(use_given_order)
            )

        if not locales and not languages and use_given_order:
            raise ValueError(
                "locales or languages must be given if use_given_order is True"
            )

        check_settings(settings)

        self._settings = settings
        self.try_previous_locales = try_previous_locales
        self.use_given_order = use_given_order
        self.languages = list(languages) if languages else None
        self.locales = locales
        self.region = region
        self.detect_languages_function = detect_languages_function
        self.previous_locales = collections.OrderedDict()

    def get_date_data(self, date_string, date_formats=None):
        """
        Parse string representing date and/or time in recognizable localized formats.
        Supports parsing multiple languages and timezones.

        :param date_string:
            A string representing date and/or time in a recognizably valid format.
        :type date_string: str
        :param date_formats:
            A list of format strings using directives as given
            `here <https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior>`_.
            The parser applies formats one by one, taking into account the detected languages.
        :type date_formats: list

        :return: a ``DateData`` object.

        :raises: ValueError - Unknown Language

        .. note:: *Period* values can be a 'day' (default), 'week', 'month', 'year', 'time'.

        *Period* represents the granularity of date parsed from the given string.

        In the example below, since no day information is present, the day is assumed to be current
        day ``16`` from *current date* (which is June 16, 2015, at the moment of writing this).
        Hence, the level of precision is ``month``:

            >>> DateDataParser().get_date_data('March 2015')
            DateData(date_obj=datetime.datetime(2015, 3, 16, 0, 0), period='month', locale='en')

        Similarly, for date strings with no day and month information present, level of precision
        is ``year`` and day ``16`` and month ``6`` are from *current_date*.

            >>> DateDataParser().get_date_data('2014')
            DateData(date_obj=datetime.datetime(2014, 6, 16, 0, 0), period='year', locale='en')

        Dates with time zone indications or UTC offsets are returned in UTC time unless
        specified using `Settings <https://timeparser.readthedocs.io/en/latest/settings.html#settings>`__.

            >>> DateDataParser().get_date_data('23 March 2000, 1:21 PM CET')
            DateData(date_obj=datetime.datetime(2000, 3, 23, 13, 21, tzinfo=<StaticTzInfo 'CET'>),
            period='day', locale='en')

        """
        if not isinstance(date_string, str):
            raise TypeError("Input type must be str")

        res = parse_with_formats(date_string, date_formats or [], self._settings)
        if res["date_obj"]:
            return res

        date_string = sanitize_date(date_string)

        for locale in self._get_applicable_locales(date_string):
            parsed_date = _DateLocaleParser.parse(
                locale, date_string, date_formats, settings=self._settings
            )
            if parsed_date:
                parsed_date["locale"] = locale.shortname
                if self.try_previous_locales:
                    self.previous_locales[locale] = None
                return parsed_date
        else:
            return DateData(date_obj=None, period="day", locale=None)

    def get_date_tuple(self, *args, **kwargs):
        date_data = self.get_date_data(*args, **kwargs)
        fields = date_data.__dict__.keys()
        date_tuple = collections.namedtuple("DateData", fields)
        return date_tuple(**date_data.__dict__)

    def get_scatex_data(self, date_string, date_formats=None):
        """
        Parse string representing date and/or time and return a SCATEX expression.
        Supports parsing multiple languages and timezones.

        :param date_string:
            A string representing date and/or time in a recognizably valid format.
        :type date_string: str
        :param date_formats:
            A list of format strings using directives as given
            `here <https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior>`_.
            The parser applies formats one by one, taking into account the detected languages.
        :type date_formats: list

        :return: a ``ScatexData`` object containing a SCATEX TemporalExpression.

        :raises: ValueError - Unknown Language

        .. note:: *Period* values can be a 'day' (default), 'week', 'month', 'year', 'time'.

        *Period* represents the granularity of date parsed from the given string.

        Example usage:
            >>> parser = DateDataParser()
            >>> result = parser.get_scatex_data('October 7, 2023')
            >>> result.scatex_expr
            Day(day=7, month=10, year=2023)
            >>> result.scatex_expr.evaluate(datetime.now())
            (datetime(2023, 10, 7, 0, 0, 0), datetime(2023, 10, 7, 23, 59, 59))
        """
        if not isinstance(date_string, str):
            raise TypeError("Input type must be str")

        # =====================================================================
        # Try composite patterns FIRST (highest priority)
        # This handles expressions like "between X and Y", "early morning"
        # =====================================================================
        from timeparser.composite_patterns import composite_parser
        composite_result = composite_parser.try_parse(date_string.strip(), self._settings)
        if composite_result:
            expr, period = composite_result
            return ScatexData(scatex_expr=expr, period=period)

        # Try custom formats
        res = parse_with_formats_scatex(date_string, date_formats or [], self._settings)
        if res["scatex_expr"]:
            return res

        date_string = sanitize_date(date_string)
        
        # Try freshness parser directly for patterns that may not pass locale 
        # applicability checks. These patterns don't need translation since they
        # use English keywords directly.
        import regex as re
        WEEKDAY_PATTERN = r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
        TIME_OF_DAY_PATTERN = r'morning|afternoon|evening|night|dawn'
        
        # Patterns that should go directly to freshness parser
        freshness_patterns = [
            r'\b(next|last)\s+\w+\b',                    # next/last X
            rf'\bthis\s+({WEEKDAY_PATTERN})\b',          # this Monday
            r'\bevery\s+\w+\b',                          # every X
            rf'^({TIME_OF_DAY_PATTERN})$',               # standalone time of day
            r'\bq[1-4](?:\s+\d{4})?\b',                  # Q1, Q2 2018
            r'\b(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b',  # ordinal quarters
            r'\b\d{1,2}(?:st|nd|rd|th)\s+century\b',     # 20th century
            r'\b(?:early|mid|late)[- ]?\d{4}s\b',        # early 1990s
            r'\b(?:since|until|before|after)\s+\d{4}\b', # since 2020
            r'\bin\s+the\s+(?:past|future)\b',           # in the past/future
            rf'\b(?:the\s+)?({TIME_OF_DAY_PATTERN})\s+of\s+', # the morning of October 7
        ]
        
        should_try_freshness = any(
            re.search(pattern, date_string, re.I) for pattern in freshness_patterns
        )
        
        if should_try_freshness:
            try:
                freshness_result = freshness_date_parser.get_scatex_data(date_string, self._settings)
                if freshness_result and freshness_result["scatex_expr"]:
                    return freshness_result
            except (OverflowError, ValueError):
                pass

        for locale in self._get_applicable_locales(date_string):
            parsed_scatex = _ScatexLocaleParser.parse(
                locale, date_string, date_formats, settings=self._settings
            )
            if parsed_scatex:
                parsed_scatex["locale"] = locale.shortname
                if self.try_previous_locales:
                    self.previous_locales[locale] = None
                return parsed_scatex
        
        return ScatexData(scatex_expr=None, period="day", locale=None)

    def _get_applicable_locales(self, date_string):
        pop_tz_cache = []

        def date_strings():
            """A generator instead of a static list to avoid calling
            pop_tz_offset_from_string if the first locale matches on unmodified
            date_string.
            """
            yield date_string
            if not pop_tz_cache:
                stripped_date_string, _ = pop_tz_offset_from_string(
                    date_string, as_offset=False
                )
                if stripped_date_string == date_string:
                    stripped_date_string = None
                pop_tz_cache[:] = [stripped_date_string]
            (stripped_date_string,) = pop_tz_cache
            if stripped_date_string is not None:
                yield stripped_date_string

        if self.try_previous_locales:
            for locale in self.previous_locales.keys():
                for s in date_strings():
                    if self._is_applicable_locale(locale, s):
                        yield locale

        if self.detect_languages_function and not self.languages and not self.locales:
            detected_languages = self.detect_languages_function(
                text=date_string,
                confidence_threshold=self._settings.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD,
            )

            self.languages = map_languages(detected_languages)

        for locale in self._get_locale_loader().get_locales(
            languages=self.languages,
            locales=self.locales,
            region=self.region,
            use_given_order=self.use_given_order,
        ):
            for s in date_strings():
                if self._is_applicable_locale(locale, s):
                    yield locale

        if self._settings.DEFAULT_LANGUAGES:
            for locale in self._get_locale_loader().get_locales(
                languages=self._settings.DEFAULT_LANGUAGES,
                locales=None,
                region=self.region,
                use_given_order=self.use_given_order,
            ):
                yield locale

    def _is_applicable_locale(self, locale, date_string):
        return locale.is_applicable(
            date_string,
            strip_timezone=False,  # it is stripped outside
            settings=self._settings,
        )

    @classmethod
    def _get_locale_loader(cls):
        if not cls.locale_loader:
            cls.locale_loader = LocaleDataLoader()
        return cls.locale_loader
