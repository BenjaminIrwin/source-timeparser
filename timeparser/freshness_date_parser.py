from datetime import datetime, time, timezone
from typing import Optional, Tuple

import regex as re
from dateutil.relativedelta import relativedelta
from tzlocal import get_localzone

from timeparser.utils import apply_timezone, localize_timezone, strip_braces
from timeparser.scatex import (
    TemporalExpression, Today, Yesterday, Tomorrow, Now,
    Shift, Period, Unit, Direction,
    Repeating, DayOfWeek, MonthOfYear, Last, Next, This,
    DayOfWeekType, MonthOfYearType, TimeOfDay, TimeOfDayType,
    Hour, Unknown,
    # Additional types for new patterns
    Year, Day, Month, Quarter, Century, Decade, Before, After, Between, Intersection,
)

from .parser import time_parser
from .timezone_parser import pop_tz_offset_from_string

_UNITS = r"decade|year|month|week|day|hour|minute|second"
PATTERN = re.compile(r"(\d+[.,]?\d*)\s*(%s)\b" % _UNITS, re.I | re.S | re.U)

# Patterns for special relative expressions
YESTERDAY_PATTERN = re.compile(r"\byesterday\b", re.I)
TOMORROW_PATTERN = re.compile(r"\btomorrow\b", re.I)
TODAY_PATTERN = re.compile(r"\btoday\b", re.I)
NOW_PATTERN = re.compile(r"\bnow\b", re.I)

# Patterns for "last X" / "next X" expressions
# Note: Use \s+ to handle multiple spaces (locale translation sometimes adds extra spaces)
LAST_PATTERN = re.compile(r"\blast\s+(\w+)\b", re.I)
NEXT_PATTERN = re.compile(r"\bnext\s+(\w+)\b", re.I)
THIS_PATTERN = re.compile(r"\bthis\s+(\w+)\b", re.I)

# Patterns for "last N weeks", "next N months" (counted expressions)
LAST_COUNT_PATTERN = re.compile(r"\blast\s+(\d+)\s+(\w+)\b", re.I)
NEXT_COUNT_PATTERN = re.compile(r"\bnext\s+(\d+)\s+(\w+)\b", re.I)

# Patterns for "every X" expressions
EVERY_PATTERN = re.compile(r"\bevery\s+(\w+)\b", re.I)

# Patterns for standalone time of day
STANDALONE_TIME_OF_DAY_PATTERN = re.compile(r"^(morning|afternoon|evening|night|dawn)$", re.I)

# Patterns for ranges: "since X", "until X", "before X", "after X"
SINCE_YEAR_PATTERN = re.compile(r"\bsince\s+(\d{4})\b", re.I)
UNTIL_YEAR_PATTERN = re.compile(r"\buntil\s+(\d{4})\b", re.I)
BEFORE_YEAR_PATTERN = re.compile(r"\bbefore\s+(\d{4})\b", re.I)
AFTER_YEAR_PATTERN = re.compile(r"\bafter\s+(\d{4})\b", re.I)
IN_THE_PAST_PATTERN = re.compile(r"\bin\s+the\s+past\b", re.I)
IN_THE_FUTURE_PATTERN = re.compile(r"\bin\s+the\s+future\b", re.I)

# Patterns for quarters
QUARTER_PATTERN = re.compile(r"\bq([1-4])(?:\s+(\d{4}))?\b", re.I)
ORDINAL_QUARTER_PATTERN = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter(?:\s+(\d{4}))?\b", re.I)

# Patterns for centuries
CENTURY_PATTERN = re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\s+century\b", re.I)

# Patterns for decade modifiers
DECADE_MODIFIER_PATTERN = re.compile(r"\b(early|mid|late)[- ]?(?:the\s+)?(\d{4})s\b", re.I)

# Patterns for "the morning of October 7" compound expressions
TIME_OF_DAY_WITH_DATE_PATTERN = re.compile(
    r"(?:on\s+|in\s+)?(?:the\s+)?(morning|afternoon|evening|night|dawn)\s+of\s+(.+)",
    re.I
)

# Patterns for translated versions of special keywords
TRANSLATED_YESTERDAY_PATTERN = re.compile(r"\b1\s+day\s+ago\b", re.I)
TRANSLATED_TOMORROW_PATTERN = re.compile(r"\bin\s+1\s+day\b", re.I)

# Patterns for "0 X ago" translated forms (from relative-type dictionary)
# These represent "now" (0 second ago), "today" (0 day ago), etc.
ZERO_SECOND_AGO_PATTERN = re.compile(r"\b0\s+second\s+ago\b", re.I)
ZERO_MINUTE_AGO_PATTERN = re.compile(r"\b0\s+minute\s+ago\b", re.I)
ZERO_DAY_AGO_PATTERN = re.compile(r"\b0\s+day\s+ago\b", re.I)
ZERO_HOUR_AGO_PATTERN = re.compile(r"\b0\s+hour\s+ago\b", re.I)
ZERO_WEEK_AGO_PATTERN = re.compile(r"\b0\s+week\s+ago\b", re.I)
ZERO_MONTH_AGO_PATTERN = re.compile(r"\b0\s+month\s+ago\b", re.I)
ZERO_YEAR_AGO_PATTERN = re.compile(r"\b0\s+year\s+ago\b", re.I)

# Pattern for "this second" which might not be translated
THIS_SECOND_PATTERN = re.compile(r"\bthis\s+second\b", re.I)

# Patterns for "2 day ago" (day before yesterday) and "in 2 day" (day after tomorrow)
TWO_DAYS_AGO_PATTERN = re.compile(r"\b2\s+day\s+ago\b", re.I)
IN_TWO_DAYS_PATTERN = re.compile(r"\bin\s+2\s+day\b", re.I)

WEEKDAY_NAMES = {
    'monday': DayOfWeekType.MONDAY,
    'tuesday': DayOfWeekType.TUESDAY,
    'wednesday': DayOfWeekType.WEDNESDAY,
    'thursday': DayOfWeekType.THURSDAY,
    'friday': DayOfWeekType.FRIDAY,
    'saturday': DayOfWeekType.SATURDAY,
    'sunday': DayOfWeekType.SUNDAY,
    'mon': DayOfWeekType.MONDAY,
    'tue': DayOfWeekType.TUESDAY,
    'wed': DayOfWeekType.WEDNESDAY,
    'thu': DayOfWeekType.THURSDAY,
    'fri': DayOfWeekType.FRIDAY,
    'sat': DayOfWeekType.SATURDAY,
    'sun': DayOfWeekType.SUNDAY,
}

MONTH_NAMES = {
    'january': MonthOfYearType.JANUARY,
    'february': MonthOfYearType.FEBRUARY,
    'march': MonthOfYearType.MARCH,
    'april': MonthOfYearType.APRIL,
    'may': MonthOfYearType.MAY,
    'june': MonthOfYearType.JUNE,
    'july': MonthOfYearType.JULY,
    'august': MonthOfYearType.AUGUST,
    'september': MonthOfYearType.SEPTEMBER,
    'october': MonthOfYearType.OCTOBER,
    'november': MonthOfYearType.NOVEMBER,
    'december': MonthOfYearType.DECEMBER,
    'jan': MonthOfYearType.JANUARY,
    'feb': MonthOfYearType.FEBRUARY,
    'mar': MonthOfYearType.MARCH,
    'apr': MonthOfYearType.APRIL,
    'jun': MonthOfYearType.JUNE,
    'jul': MonthOfYearType.JULY,
    'aug': MonthOfYearType.AUGUST,
    'sep': MonthOfYearType.SEPTEMBER,
    'oct': MonthOfYearType.OCTOBER,
    'nov': MonthOfYearType.NOVEMBER,
    'dec': MonthOfYearType.DECEMBER,
}

TIME_OF_DAY_NAMES = {
    'morning': TimeOfDayType.MORNING,
    'afternoon': TimeOfDayType.AFTERNOON,
    'evening': TimeOfDayType.EVENING,
    'night': TimeOfDayType.NIGHT,
    'noon': TimeOfDayType.NOON,
    'midnight': TimeOfDayType.MIDNIGHT,
    'dawn': TimeOfDayType.DAWN,
}

UNIT_NAMES = {
    'second': Unit.SECOND,
    'seconds': Unit.SECOND,
    'minute': Unit.MINUTE,
    'minutes': Unit.MINUTE,
    'hour': Unit.HOUR,
    'hours': Unit.HOUR,
    'day': Unit.DAY,
    'days': Unit.DAY,
    'week': Unit.WEEK,
    'weeks': Unit.WEEK,
    'month': Unit.MONTH,
    'months': Unit.MONTH,
    'year': Unit.YEAR,
    'years': Unit.YEAR,
    'decade': Unit.DECADE,
    'decades': Unit.DECADE,
}


class FreshnessDateDataParser:
    """Parses date string like "1 year, 2 months ago" and "3 hours, 50 minutes ago" """

    def _are_all_words_units(self, date_string):
        skip = [_UNITS, r"ago|in|\d+", r":|[ap]m"]

        date_string = re.sub(r"\s+", " ", date_string.strip())

        words = [x for x in re.split(r"\W", date_string) if x]
        words = [x for x in words if not re.match(r"%s" % "|".join(skip), x)]
        return not words

    def _parse_time(self, date_string, settings):
        """Attempts to parse time part of date strings like '1 day ago, 2 PM'"""
        date_string = PATTERN.sub("", date_string)
        date_string = re.sub(r"\b(?:ago|in)\b", "", date_string)
        try:
            return time_parser(date_string)
        except Exception:
            pass

    def get_local_tz(self):
        return get_localzone()

    def parse(self, date_string, settings):
        date_string = strip_braces(date_string)
        date_string, ptz = pop_tz_offset_from_string(date_string)
        _time = self._parse_time(date_string, settings)

        _settings_tz = settings.TIMEZONE.lower()

        def apply_time(dateobj, timeobj):
            if not isinstance(_time, time):
                return dateobj

            return dateobj.replace(
                hour=timeobj.hour,
                minute=timeobj.minute,
                second=timeobj.second,
                microsecond=timeobj.microsecond,
            )

        if settings.RELATIVE_BASE:
            now = settings.RELATIVE_BASE

            if "local" not in _settings_tz:
                now = localize_timezone(now, settings.TIMEZONE)

            if ptz:
                if now.tzinfo:
                    now = now.astimezone(ptz)
                else:
                    if hasattr(ptz, "localize"):
                        now = ptz.localize(now)
                    else:
                        now = now.replace(tzinfo=ptz)

            if not now.tzinfo:
                now = now.replace(tzinfo=self.get_local_tz())

        elif ptz:
            localized_now = datetime.now(ptz)

            if "local" in _settings_tz:
                now = localized_now
            else:
                now = apply_timezone(localized_now, settings.TIMEZONE)

        else:
            if "local" not in _settings_tz:
                utc_dt = datetime.now(tz=timezone.utc)
                now = apply_timezone(utc_dt, settings.TIMEZONE)
            else:
                now = datetime.now(self.get_local_tz())

        date, period = self._parse_date(date_string, now, settings.PREFER_DATES_FROM)

        if date:
            old_date = date
            date = apply_time(date, _time)
            if settings.RETURN_TIME_AS_PERIOD and old_date != date:
                period = "time"

            if settings.TO_TIMEZONE:
                date = apply_timezone(date, settings.TO_TIMEZONE)

            if not settings.RETURN_AS_TIMEZONE_AWARE or (
                settings.RETURN_AS_TIMEZONE_AWARE
                and "default" == settings.RETURN_AS_TIMEZONE_AWARE
                and not ptz
            ):
                date = date.replace(tzinfo=None)

        return date, period

    def _parse_date(self, date_string, now, prefer_dates_from):
        if not self._are_all_words_units(date_string):
            return None, None

        kwargs = self.get_kwargs(date_string)
        if not kwargs:
            return None, None
        period = "day"
        if "days" not in kwargs:
            for k in ["weeks", "months", "years"]:
                if k in kwargs:
                    period = k[:-1]
                    break
        td = relativedelta(**kwargs)

        if (
            re.search(r"\bin\b", date_string)
            or re.search(r"\bfuture\b", prefer_dates_from)
            and not re.search(r"\bago\b", date_string)
        ):
            date = now + td
        else:
            date = now - td
        return date, period

    def get_kwargs(self, date_string):
        m = PATTERN.findall(date_string)
        if not m:
            return {}

        kwargs = {}
        for num, unit in m:
            kwargs[unit + "s"] = float(num.replace(",", "."))
        if "decades" in kwargs:
            kwargs["years"] = 10 * kwargs["decades"] + kwargs.get("years", 0)
            del kwargs["decades"]
        return kwargs

    def get_date_data(self, date_string, settings=None):
        from timeparser.date import DateData

        date, period = self.parse(date_string, settings)
        return DateData(date_obj=date, period=period)

    def parse_scatex(self, date_string: str, settings) -> Tuple[Optional[TemporalExpression], Optional[str]]:
        """
        Parse a relative date string and return a SCATEX expression.
        
        :param date_string: The date string to parse (e.g., "3 days ago", "next Monday")
        :param settings: Parser settings
        :return: Tuple of (SCATEX expression, period)
        """
        date_string = strip_braces(date_string)
        date_string_clean, _ = pop_tz_offset_from_string(date_string)
        
        # Normalize multiple spaces to single space (locale translation can add extra spaces)
        date_string_clean = re.sub(r'\s+', ' ', date_string_clean).strip()
        
        # =====================================================================
        # Try composite patterns FIRST (highest priority)
        # This handles expressions like "between X and Y", "early morning of X"
        # =====================================================================
        from .composite_patterns import composite_parser
        composite_result = composite_parser.try_parse(date_string_clean, settings)
        if composite_result:
            return composite_result
        
        # Check for special keywords first (including translated versions)
        if NOW_PATTERN.search(date_string_clean):
            return (Now(), "time")
        
        # Check for translated "0 second ago" or "this second" which means "now"
        if ZERO_SECOND_AGO_PATTERN.search(date_string_clean) or THIS_SECOND_PATTERN.search(date_string_clean):
            return (Now(), "time")
        
        # Check for "0 minute ago" which means "this minute"
        if ZERO_MINUTE_AGO_PATTERN.search(date_string_clean):
            return (This(interval=Repeating(unit=Unit.MINUTE)), "minute")
        
        if TODAY_PATTERN.search(date_string_clean):
            return (Today(), "day")
        
        # Check for translated "0 day ago" which means "today"
        if ZERO_DAY_AGO_PATTERN.search(date_string_clean):
            return (Today(), "day")
        
        # Check for "0 hour ago" which means "this hour"
        if ZERO_HOUR_AGO_PATTERN.search(date_string_clean):
            return (This(interval=Repeating(unit=Unit.HOUR)), "hour")
        
        # Check for "0 week ago" which means "this week"
        if ZERO_WEEK_AGO_PATTERN.search(date_string_clean):
            return (This(interval=Repeating(unit=Unit.WEEK)), "week")
        
        # Check for "0 month ago" which means "this month"
        if ZERO_MONTH_AGO_PATTERN.search(date_string_clean):
            return (This(interval=Repeating(unit=Unit.MONTH)), "month")
        
        # Check for "0 year ago" which means "this year"
        if ZERO_YEAR_AGO_PATTERN.search(date_string_clean):
            return (This(interval=Repeating(unit=Unit.YEAR)), "year")
        
        # Check both original and translated versions of yesterday/tomorrow
        if YESTERDAY_PATTERN.search(date_string_clean):
            return (Yesterday(), "day")
        
        # Check for translated "1 day ago" which means yesterday
        if TRANSLATED_YESTERDAY_PATTERN.search(date_string_clean):
            return (Yesterday(), "day")
        
        if TOMORROW_PATTERN.search(date_string_clean):
            return (Tomorrow(), "day")
        
        # Check for translated "in 1 day" which means tomorrow
        if TRANSLATED_TOMORROW_PATTERN.search(date_string_clean):
            return (Tomorrow(), "day")
        
        # Check for "2 day ago" (day before yesterday)
        if TWO_DAYS_AGO_PATTERN.search(date_string_clean):
            return (Shift(
                interval=Today(),
                period=Period(unit=Unit.DAY, value=2),
                direction=Direction.BEFORE
            ), "day")
        
        # Check for "in 2 day" (day after tomorrow)
        if IN_TWO_DAYS_PATTERN.search(date_string_clean):
            return (Shift(
                interval=Today(),
                period=Period(unit=Unit.DAY, value=2),
                direction=Direction.AFTER
            ), "day")
        
        # =====================================================================
        # NEW PATTERNS: Standalone time of day, quarters, centuries, etc.
        # =====================================================================
        
        # Check for standalone time of day ("morning", "afternoon", etc.)
        tod_match = STANDALONE_TIME_OF_DAY_PATTERN.match(date_string_clean)
        if tod_match:
            tod_name = tod_match.group(1).lower()
            if tod_name in TIME_OF_DAY_NAMES:
                return (TimeOfDay(type=TIME_OF_DAY_NAMES[tod_name]), "time")
        
        # Check for "the morning of October 7" compound expressions
        tod_date_match = TIME_OF_DAY_WITH_DATE_PATTERN.match(date_string_clean)
        if tod_date_match:
            tod_name = tod_date_match.group(1).lower()
            date_part = tod_date_match.group(2).strip()
            if tod_name in TIME_OF_DAY_NAMES:
                # Parse the date part using the main parser
                from timeparser import parse as timeparser_parse
                date_expr = timeparser_parse(date_part)
                if date_expr and not isinstance(date_expr, Unknown):
                    # Wrap in Intersection with TimeOfDay
                    return (Intersection(
                        intervals=[date_expr, TimeOfDay(type=TIME_OF_DAY_NAMES[tod_name])]
                    ), "time")
        
        # Check for quarters ("Q2 2018", "Q3")
        quarter_match = QUARTER_PATTERN.match(date_string_clean)
        if quarter_match:
            q_num = int(quarter_match.group(1))
            q_year = int(quarter_match.group(2)) if quarter_match.group(2) else None
            return (Quarter(quarter=q_num, year=q_year), "quarter")
        
        # Check for ordinal quarters ("first quarter 2020", "second quarter")
        ord_quarter_match = ORDINAL_QUARTER_PATTERN.match(date_string_clean)
        if ord_quarter_match:
            ordinal = ord_quarter_match.group(1).lower()
            ordinal_map = {'first': 1, '1st': 1, 'second': 2, '2nd': 2, 
                           'third': 3, '3rd': 3, 'fourth': 4, '4th': 4}
            q_num = ordinal_map.get(ordinal, 1)
            q_year = int(ord_quarter_match.group(2)) if ord_quarter_match.group(2) else None
            return (Quarter(quarter=q_num, year=q_year), "quarter")
        
        # Check for centuries ("20th century", "21st century")
        century_match = CENTURY_PATTERN.match(date_string_clean)
        if century_match:
            c_num = int(century_match.group(1))
            return (Century(number=c_num), "century")
        
        # Check for decade modifiers ("early 1990s", "mid-2000s", "late 2010s")
        decade_mod_match = DECADE_MODIFIER_PATTERN.match(date_string_clean)
        if decade_mod_match:
            modifier = decade_mod_match.group(1).lower()
            decade_start = int(decade_mod_match.group(2))
            if modifier == 'early':
                return (Between(start_interval=Year(digits=decade_start), 
                               end_interval=Year(digits=decade_start + 4)), "decade")
            elif modifier == 'mid':
                return (Between(start_interval=Year(digits=decade_start + 4), 
                               end_interval=Year(digits=decade_start + 6)), "decade")
            elif modifier == 'late':
                return (Between(start_interval=Year(digits=decade_start + 6), 
                               end_interval=Year(digits=decade_start + 9)), "decade")
        
        # Check for "since YEAR", "until YEAR", "before YEAR", "after YEAR"
        since_match = SINCE_YEAR_PATTERN.match(date_string_clean)
        if since_match:
            year = int(since_match.group(1))
            return (After(interval=Year(digits=year)), "year")
        
        until_match = UNTIL_YEAR_PATTERN.match(date_string_clean)
        if until_match:
            year = int(until_match.group(1))
            return (Before(interval=Year(digits=year)), "year")
        
        before_match = BEFORE_YEAR_PATTERN.match(date_string_clean)
        if before_match:
            year = int(before_match.group(1))
            return (Before(interval=Year(digits=year)), "year")
        
        after_match = AFTER_YEAR_PATTERN.match(date_string_clean)
        if after_match:
            year = int(after_match.group(1))
            return (After(interval=Year(digits=year)), "year")
        
        # Check for "in the past" / "in the future"
        if IN_THE_PAST_PATTERN.match(date_string_clean):
            return (Before(interval=Now()), "day")
        
        if IN_THE_FUTURE_PATTERN.match(date_string_clean):
            return (After(interval=Now()), "day")
        
        # Check for "every X" pattern
        every_match = EVERY_PATTERN.match(date_string_clean)
        if every_match:
            unit_name = every_match.group(1).lower()
            # Check if it's a weekday
            if unit_name in WEEKDAY_NAMES:
                return (DayOfWeek(type=WEEKDAY_NAMES[unit_name]), "day")
            # Check if it's a unit
            if unit_name in UNIT_NAMES:
                return (Repeating(unit=UNIT_NAMES[unit_name]), unit_name)
        
        # Check for "last N weeks", "next N months" (counted expressions)
        last_count_match = LAST_COUNT_PATTERN.match(date_string_clean)
        if last_count_match:
            count = int(last_count_match.group(1))
            unit_name = last_count_match.group(2).lower()
            if unit_name in UNIT_NAMES or unit_name.rstrip('s') in UNIT_NAMES:
                # Normalize unit name (handle both "week" and "weeks")
                unit_key = unit_name if unit_name in UNIT_NAMES else unit_name.rstrip('s')
                if unit_key in UNIT_NAMES:
                    return (Last(interval=Repeating(unit=UNIT_NAMES[unit_key]), count=count), unit_key)
        
        next_count_match = NEXT_COUNT_PATTERN.match(date_string_clean)
        if next_count_match:
            count = int(next_count_match.group(1))
            unit_name = next_count_match.group(2).lower()
            if unit_name in UNIT_NAMES or unit_name.rstrip('s') in UNIT_NAMES:
                unit_key = unit_name if unit_name in UNIT_NAMES else unit_name.rstrip('s')
                if unit_key in UNIT_NAMES:
                    return (Next(interval=Repeating(unit=UNIT_NAMES[unit_key]), count=count), unit_key)
        
        # =====================================================================
        # END NEW PATTERNS
        # =====================================================================
        
        # Check for "last X" pattern
        last_match = LAST_PATTERN.search(date_string_clean)
        if last_match:
            unit_name = last_match.group(1).lower()
            expr = self._build_last_next_scatex(unit_name, is_last=True)
            if expr:
                period = self._get_period_for_unit(unit_name)
                return (expr, period)
        
        # Check for "next X" pattern
        next_match = NEXT_PATTERN.search(date_string_clean)
        if next_match:
            unit_name = next_match.group(1).lower()
            expr = self._build_last_next_scatex(unit_name, is_last=False)
            if expr:
                period = self._get_period_for_unit(unit_name)
                return (expr, period)
        
        # Check for "this X" pattern
        this_match = THIS_PATTERN.search(date_string_clean)
        if this_match:
            unit_name = this_match.group(1).lower()
            expr = self._build_this_scatex(unit_name)
            if expr:
                period = self._get_period_for_unit(unit_name)
                return (expr, period)
        
        # Parse relative duration expressions like "3 days ago", "in 2 weeks"
        if not self._are_all_words_units(date_string_clean):
            return (None, None)
        
        kwargs = self.get_kwargs(date_string_clean)
        if not kwargs:
            return (None, None)
        
        # Build SCATEX Shift expression
        scatex_expr, period = self._build_shift_scatex(date_string_clean, kwargs, settings)
        return (scatex_expr, period)
    
    def _build_last_next_scatex(self, unit_name: str, is_last: bool) -> Optional[TemporalExpression]:
        """Build Last() or Next() SCATEX expression for 'last X' / 'next X' patterns."""
        wrapper = Last if is_last else Next
        
        # Check if it's a weekday
        if unit_name in WEEKDAY_NAMES:
            return wrapper(interval=DayOfWeek(type=WEEKDAY_NAMES[unit_name]))
        
        # Check if it's a month
        if unit_name in MONTH_NAMES:
            return wrapper(interval=MonthOfYear(type=MONTH_NAMES[unit_name]))
        
        # Check if it's a time of day
        if unit_name in TIME_OF_DAY_NAMES:
            return wrapper(interval=TimeOfDay(type=TIME_OF_DAY_NAMES[unit_name]))
        
        # Check if it's a unit (week, month, year, etc.)
        if unit_name in UNIT_NAMES:
            return wrapper(interval=Repeating(unit=UNIT_NAMES[unit_name]))
        
        return None
    
    def _build_this_scatex(self, unit_name: str) -> Optional[TemporalExpression]:
        """Build This() SCATEX expression for 'this X' patterns."""
        # Check if it's a weekday
        if unit_name in WEEKDAY_NAMES:
            return This(interval=DayOfWeek(type=WEEKDAY_NAMES[unit_name]))
        
        # Check if it's a month
        if unit_name in MONTH_NAMES:
            return This(interval=MonthOfYear(type=MONTH_NAMES[unit_name]))
        
        # Check if it's a time of day
        if unit_name in TIME_OF_DAY_NAMES:
            return This(interval=TimeOfDay(type=TIME_OF_DAY_NAMES[unit_name]))
        
        # Check if it's a unit (week, month, year, etc.)
        if unit_name in UNIT_NAMES:
            return This(interval=Repeating(unit=UNIT_NAMES[unit_name]))
        
        return None
    
    def _build_shift_scatex(self, date_string: str, kwargs: dict, settings) -> Tuple[Optional[TemporalExpression], str]:
        """
        Build a Shift SCATEX expression from parsed duration kwargs.
        
        Examples:
            "3 days ago" → Shift(interval=Today(), period=Period(unit=DAY, value=3), direction=BEFORE)
            "in 2 weeks" → Shift(interval=Today(), period=Period(unit=WEEK, value=2), direction=AFTER)
        """
        # Determine direction
        is_future = (
            re.search(r"\bin\b", date_string)
            or (re.search(r"\bfuture\b", settings.PREFER_DATES_FROM) and not re.search(r"\bago\b", date_string))
        )
        direction = Direction.AFTER if is_future else Direction.BEFORE
        
        # Determine period from the largest unit
        period = "day"
        if "days" not in kwargs:
            for k in ["weeks", "months", "years"]:
                if k in kwargs:
                    period = k[:-1]
                    break
        
        # Build the shift expression
        # For multiple units, we need to chain shifts or combine periods
        # For simplicity, we'll use the primary (largest) unit
        
        # Map kwargs to SCATEX units and build Period objects
        unit_priority = ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds']
        
        # Find the primary unit (largest)
        primary_unit = None
        primary_value = 0
        for unit_name in unit_priority:
            if unit_name in kwargs:
                primary_unit = unit_name
                primary_value = int(kwargs[unit_name])
                break
        
        if primary_unit is None:
            return (None, None)
        
        # Map to SCATEX Unit
        unit_map = {
            'seconds': Unit.SECOND,
            'minutes': Unit.MINUTE,
            'hours': Unit.HOUR,
            'days': Unit.DAY,
            'weeks': Unit.WEEK,
            'months': Unit.MONTH,
            'years': Unit.YEAR,
        }
        
        scatex_unit = unit_map.get(primary_unit, Unit.DAY)
        
        # Build the Shift expression
        # Start with Today() as the base interval
        base_interval = Today()
        
        # If there are multiple units, we build nested shifts
        # Start from the smallest unit and work up
        result_expr = base_interval
        
        for unit_name in reversed(unit_priority):
            if unit_name in kwargs and kwargs[unit_name] > 0:
                scatex_unit = unit_map.get(unit_name, Unit.DAY)
                value = int(kwargs[unit_name])
                result_expr = Shift(
                    interval=result_expr,
                    period=Period(unit=scatex_unit, value=value),
                    direction=direction
                )
        
        return (result_expr, period)
    
    def _get_period_for_unit(self, unit_name: str) -> str:
        """Get the period string for a unit name."""
        if unit_name in WEEKDAY_NAMES:
            return "day"
        if unit_name in MONTH_NAMES:
            return "month"
        if unit_name in TIME_OF_DAY_NAMES:
            return "time"
        if unit_name in ['year', 'years']:
            return "year"
        if unit_name in ['month', 'months']:
            return "month"
        if unit_name in ['week', 'weeks']:
            return "week"
        return "day"
    
    def get_scatex_data(self, date_string: str, settings=None):
        """
        Parse a relative date string and return ScatexData.
        
        :param date_string: The date string to parse
        :param settings: Parser settings
        :return: ScatexData object with scatex_expr
        """
        from timeparser.date import ScatexData
        
        scatex_expr, period = self.parse_scatex(date_string, settings)
        return ScatexData(scatex_expr=scatex_expr, period=period)


freshness_date_parser = FreshnessDateDataParser()
