"""
Tests for SCATEX (SCATE eXecutable) output from timeparser.

These tests verify that timeparser correctly parses date strings and returns
SCATEX compositional expressions instead of datetime objects.
"""

import pytest
from datetime import datetime

import timeparser
from timeparser.scatex import (
    TemporalExpression,
    Year, Month, Day, Hour, Minute, Second,
    DayOfWeek, DayOfWeekType, MonthOfYear, MonthOfYearType,
    Repeating, Unit, Direction,
    This, Last, Next, Shift, Period,
    Today, Yesterday, Tomorrow, Now, Unknown,
)


class TestAbsoluteDates:
    """Tests for parsing absolute date strings."""
    
    def test_full_date(self):
        """Test parsing a complete date (day, month, year)."""
        result = timeparser.parse("October 7, 2023", languages=['en'])
        assert isinstance(result, Day)
        assert result.day == 7
        assert result.month == 10
        assert result.year == 2023
    
    def test_month_year(self):
        """Test parsing a month and year (no day)."""
        result = timeparser.parse("March 2015", languages=['en'])
        assert isinstance(result, Month)
        assert result.month == 3
        assert result.year == 2015
    
    def test_year_only(self):
        """Test parsing just a year."""
        result = timeparser.parse("2014", languages=['en'])
        assert isinstance(result, Year)
        assert result.digits == 2014
    
    def test_partial_date_no_year(self):
        """Test parsing a date without year (partial)."""
        result = timeparser.parse("October 7", languages=['en'])
        assert isinstance(result, Day)
        assert result.day == 7
        assert result.month == 10
        assert result.year is None  # Partial - year unknown
    
    def test_time_only(self):
        """Test parsing just a time."""
        result = timeparser.parse("15:30", languages=['en'])
        assert isinstance(result, Minute)
        assert result.hour == 15
        assert result.minute == 30


class TestRelativeDates:
    """Tests for parsing relative date strings."""
    
    def test_days_ago(self):
        """Test parsing '3 days ago'."""
        result = timeparser.parse("3 days ago", languages=['en'])
        assert isinstance(result, Shift)
        assert isinstance(result.interval, Today)
        assert result.period.unit == Unit.DAY
        assert result.period.value == 3
        assert result.direction == Direction.BEFORE
    
    def test_yesterday(self):
        """Test parsing 'yesterday'."""
        result = timeparser.parse("yesterday", languages=['en'])
        assert isinstance(result, Yesterday)
    
    def test_tomorrow(self):
        """Test parsing 'tomorrow'."""
        result = timeparser.parse("tomorrow", languages=['en'])
        assert isinstance(result, Tomorrow)
    
    def test_today(self):
        """Test parsing 'today'."""
        result = timeparser.parse("today", languages=['en'])
        assert isinstance(result, Today)
    
    def test_now(self):
        """Test parsing 'now'."""
        result = timeparser.parse("now", languages=['en'])
        assert isinstance(result, Now)


class TestNextLastThis:
    """Tests for next/last/this patterns."""
    
    def test_next_monday(self):
        """Test parsing 'next Monday'."""
        result = timeparser.parse("next Monday", languages=['en'])
        assert isinstance(result, Next)
        assert isinstance(result.interval, DayOfWeek)
        assert result.interval.type == DayOfWeekType.MONDAY
    
    def test_last_friday(self):
        """Test parsing 'last Friday'."""
        result = timeparser.parse("last Friday", languages=['en'])
        assert isinstance(result, Last)
        assert isinstance(result.interval, DayOfWeek)
        assert result.interval.type == DayOfWeekType.FRIDAY
    
    def test_last_week(self):
        """Test parsing 'last week'."""
        result = timeparser.parse("last week", languages=['en'])
        assert isinstance(result, Last)
        assert isinstance(result.interval, Repeating)
        assert result.interval.unit == Unit.WEEK
    
    def test_this_week(self):
        """Test parsing 'this week'."""
        result = timeparser.parse("this week", languages=['en'])
        assert isinstance(result, This)
        assert isinstance(result.interval, Repeating)
        assert result.interval.unit == Unit.WEEK
    
    def test_last_month(self):
        """Test parsing 'last month'."""
        result = timeparser.parse("last month", languages=['en'])
        assert isinstance(result, Last)
        assert isinstance(result.interval, Repeating)
        assert result.interval.unit == Unit.MONTH
    
    def test_this_year(self):
        """Test parsing 'this year'."""
        result = timeparser.parse("this year", languages=['en'])
        assert isinstance(result, This)
        assert isinstance(result.interval, Repeating)
        assert result.interval.unit == Unit.YEAR


class TestEvaluation:
    """Tests for evaluating SCATEX expressions with an anchor date."""
    
    @pytest.fixture
    def anchor(self):
        """Provide a fixed anchor date for testing."""
        return datetime(2023, 10, 15, 12, 0, 0)
    
    def test_evaluate_absolute_date(self, anchor):
        """Test evaluating an absolute date."""
        result = timeparser.parse("October 7, 2023", languages=['en'])
        start, end = result.evaluate(anchor)
        assert start == datetime(2023, 10, 7, 0, 0, 0)
        assert end == datetime(2023, 10, 7, 23, 59, 59)
    
    def test_evaluate_days_ago(self, anchor):
        """Test evaluating '3 days ago'."""
        result = timeparser.parse("3 days ago", languages=['en'])
        start, end = result.evaluate(anchor)
        # 3 days before Oct 15 = Oct 12
        assert start == datetime(2023, 10, 12, 0, 0, 0)
        assert end == datetime(2023, 10, 12, 23, 59, 59)
    
    def test_evaluate_yesterday(self, anchor):
        """Test evaluating 'yesterday'."""
        result = timeparser.parse("yesterday", languages=['en'])
        start, end = result.evaluate(anchor)
        # Yesterday from Oct 15 = Oct 14
        assert start == datetime(2023, 10, 14, 0, 0, 0)
        assert end == datetime(2023, 10, 14, 23, 59, 59)
    
    def test_evaluate_tomorrow(self, anchor):
        """Test evaluating 'tomorrow'."""
        result = timeparser.parse("tomorrow", languages=['en'])
        start, end = result.evaluate(anchor)
        # Tomorrow from Oct 15 = Oct 16
        assert start == datetime(2023, 10, 16, 0, 0, 0)
        assert end == datetime(2023, 10, 16, 23, 59, 59)
    
    def test_evaluate_next_monday(self, anchor):
        """Test evaluating 'next Monday'."""
        result = timeparser.parse("next Monday", languages=['en'])
        start, end = result.evaluate(anchor)
        # Oct 15, 2023 is Sunday, so next Monday is Oct 16
        assert start == datetime(2023, 10, 16, 0, 0, 0)
        assert end == datetime(2023, 10, 16, 23, 59, 59)
    
    def test_evaluate_partial_date(self, anchor):
        """Test that partial dates cannot be evaluated without anchor year."""
        result = timeparser.parse("October 7", languages=['en'])
        assert isinstance(result, Day)
        assert result.year is None
        # Evaluation should return (None, None) for partial dates
        start, end = result.evaluate(anchor)
        assert start is None
        assert end is None


class TestMultiLanguage:
    """Tests for multi-language support."""
    
    def test_spanish_days_ago(self):
        """Test parsing Spanish relative date."""
        result = timeparser.parse("hace 3 días", languages=['es'])
        assert isinstance(result, Shift)
        assert result.period.unit == Unit.DAY
        assert result.period.value == 3
        assert result.direction == Direction.BEFORE
    
    def test_french_date(self):
        """Test parsing French date."""
        result = timeparser.parse("7 octobre 2023", languages=['fr'])
        assert isinstance(result, Day)
        assert result.day == 7
        assert result.month == 10
        assert result.year == 2023


class TestEdgeCases:
    """Tests for edge cases and special handling."""
    
    def test_none_for_unparseable(self):
        """Test that unparseable strings return None."""
        result = timeparser.parse("this is not a date")
        assert result is None
    
    def test_empty_string(self):
        """Test that empty strings return None."""
        result = timeparser.parse("")
        assert result is None
    
    def test_scatex_data_class(self):
        """Test the ScatexData wrapper class."""
        from timeparser.date import DateDataParser
        
        parser = DateDataParser(languages=['en'])
        data = parser.get_scatex_data("October 7, 2023")
        
        assert data.scatex_expr is not None
        assert data.period == "day"
        assert data.locale == "en"
        
        # Test evaluate method on ScatexData
        anchor = datetime(2023, 10, 15)
        start, end = data.evaluate(anchor)
        assert start == datetime(2023, 10, 7, 0, 0, 0)


class TestNewPatternVariants:
    """Tests for newly added pattern variants (informal, abbreviations, etc.)."""
    
    # --- NOW variants ---
    
    def test_right_now(self):
        """Test parsing 'right now'."""
        result = timeparser.parse("right now", languages=['en'])
        assert isinstance(result, Now)
    
    def test_just_now(self):
        """Test parsing 'just now'."""
        result = timeparser.parse("just now", languages=['en'])
        assert isinstance(result, Now)
    
    def test_rn_abbreviation(self):
        """Test parsing 'rn' (abbreviation for right now)."""
        result = timeparser.parse("rn", languages=['en'])
        assert isinstance(result, Now)
    
    def test_atm_abbreviation(self):
        """Test parsing 'atm' (abbreviation for at the moment)."""
        result = timeparser.parse("atm", languages=['en'])
        assert isinstance(result, Now)
    
    def test_currently(self):
        """Test parsing 'currently'."""
        result = timeparser.parse("currently", languages=['en'])
        assert isinstance(result, Now)
    
    def test_presently(self):
        """Test parsing 'presently'."""
        result = timeparser.parse("presently", languages=['en'])
        assert isinstance(result, Now)
    
    def test_at_the_moment(self):
        """Test parsing 'at the moment'."""
        result = timeparser.parse("at the moment", languages=['en'])
        assert isinstance(result, Now)
    
    def test_as_we_speak(self):
        """Test parsing 'as we speak'."""
        result = timeparser.parse("as we speak", languages=['en'])
        assert isinstance(result, Now)
    
    # --- TODAY variants ---
    
    def test_2day_abbreviation(self):
        """Test parsing '2day' (abbreviation for today)."""
        result = timeparser.parse("2day", languages=['en'])
        assert isinstance(result, Today)
    
    def test_tdy_abbreviation(self):
        """Test parsing 'tdy' (abbreviation for today)."""
        result = timeparser.parse("tdy", languages=['en'])
        assert isinstance(result, Today)
    
    def test_earlier_today(self):
        """Test parsing 'earlier today'."""
        result = timeparser.parse("earlier today", languages=['en'])
        assert isinstance(result, Today)
    
    def test_as_of_today(self):
        """Test parsing 'as of today'."""
        result = timeparser.parse("as of today", languages=['en'])
        assert isinstance(result, Today)
    
    def test_by_cob(self):
        """Test parsing 'by COB' (close of business)."""
        result = timeparser.parse("by COB", languages=['en'])
        assert isinstance(result, Today)
    
    def test_this_day(self):
        """Test parsing 'this day'."""
        result = timeparser.parse("this day", languages=['en'])
        assert isinstance(result, Today)
    
    def test_on_this_day(self):
        """Test parsing 'on this day'."""
        result = timeparser.parse("on this day", languages=['en'])
        assert isinstance(result, Today)
    
    # --- YESTERDAY variants ---
    
    def test_yday_abbreviation(self):
        """Test parsing 'yday' (abbreviation for yesterday)."""
        result = timeparser.parse("yday", languages=['en'])
        assert isinstance(result, Yesterday)
    
    def test_yest_abbreviation(self):
        """Test parsing 'yest' (abbreviation for yesterday)."""
        result = timeparser.parse("yest", languages=['en'])
        assert isinstance(result, Yesterday)
    
    def test_yesterday_morning(self):
        """Test parsing 'yesterday morning'."""
        result = timeparser.parse("yesterday morning", languages=['en'])
        assert isinstance(result, Yesterday)
    
    def test_the_day_before(self):
        """Test parsing 'the day before'."""
        result = timeparser.parse("the day before", languages=['en'])
        assert isinstance(result, Yesterday)
    
    def test_a_day_ago(self):
        """Test parsing 'a day ago'."""
        result = timeparser.parse("a day ago", languages=['en'])
        assert isinstance(result, Yesterday)
    
    # --- TOMORROW variants ---
    
    def test_tmrw_abbreviation(self):
        """Test parsing 'tmrw' (abbreviation for tomorrow)."""
        result = timeparser.parse("tmrw", languages=['en'])
        assert isinstance(result, Tomorrow)
    
    def test_2moro_abbreviation(self):
        """Test parsing '2moro' (abbreviation for tomorrow)."""
        result = timeparser.parse("2moro", languages=['en'])
        assert isinstance(result, Tomorrow)
    
    def test_tomo_abbreviation(self):
        """Test parsing 'tomo' (abbreviation for tomorrow)."""
        result = timeparser.parse("tomo", languages=['en'])
        assert isinstance(result, Tomorrow)
    
    def test_the_morrow(self):
        """Test parsing 'the morrow'."""
        result = timeparser.parse("the morrow", languages=['en'])
        assert isinstance(result, Tomorrow)
    
    def test_tomorrow_morning(self):
        """Test parsing 'tomorrow morning'."""
        result = timeparser.parse("tomorrow morning", languages=['en'])
        assert isinstance(result, Tomorrow)
    
    def test_in_a_day(self):
        """Test parsing 'in a day'."""
        result = timeparser.parse("in a day", languages=['en'])
        assert isinstance(result, Tomorrow)
    
    # --- Day before yesterday / Day after tomorrow ---
    
    def test_day_before_yesterday(self):
        """Test parsing 'the day before yesterday'."""
        result = timeparser.parse("the day before yesterday", languages=['en'])
        assert isinstance(result, Shift)
        assert result.direction == Direction.BEFORE
        assert result.period.value == 2
        assert result.period.unit == Unit.DAY
    
    def test_two_days_ago(self):
        """Test parsing 'two days ago'."""
        result = timeparser.parse("two days ago", languages=['en'])
        assert isinstance(result, Shift)
        assert result.direction == Direction.BEFORE
        assert result.period.value == 2
    
    def test_day_after_tomorrow(self):
        """Test parsing 'day after tomorrow'."""
        result = timeparser.parse("day after tomorrow", languages=['en'])
        assert isinstance(result, Shift)
        assert result.direction == Direction.AFTER
        assert result.period.value == 2
        assert result.period.unit == Unit.DAY
    
    def test_in_two_days(self):
        """Test parsing 'in two days'."""
        result = timeparser.parse("in two days", languages=['en'])
        assert isinstance(result, Shift)
        assert result.direction == Direction.AFTER
        assert result.period.value == 2
    
    # --- THIS variants ---
    
    def test_this_hour(self):
        """Test parsing 'this hour'."""
        result = timeparser.parse("this hour", languages=['en'])
        assert isinstance(result, This)
        assert isinstance(result.interval, Repeating)
        assert result.interval.unit == Unit.HOUR
    
    def test_this_minute(self):
        """Test parsing 'this minute'."""
        result = timeparser.parse("this minute", languages=['en'])
        assert isinstance(result, This)
        assert isinstance(result.interval, Repeating)
        assert result.interval.unit == Unit.MINUTE
    
    def test_this_second_is_now(self):
        """Test parsing 'this second' returns Now."""
        result = timeparser.parse("this second", languages=['en'])
        assert isinstance(result, Now)


class TestThisWeekday:
    """Tests for 'this [weekday]' patterns."""
    
    def test_this_monday(self):
        """Test parsing 'this Monday'."""
        result = timeparser.parse("this Monday", languages=['en'])
        assert isinstance(result, This)
        assert isinstance(result.interval, DayOfWeek)
        assert result.interval.type == DayOfWeekType.MONDAY
    
    def test_this_friday(self):
        """Test parsing 'this Friday'."""
        result = timeparser.parse("this Friday", languages=['en'])
        assert isinstance(result, This)
        assert isinstance(result.interval, DayOfWeek)
        assert result.interval.type == DayOfWeekType.FRIDAY
    
    def test_this_saturday(self):
        """Test parsing 'this Saturday'."""
        result = timeparser.parse("this Saturday", languages=['en'])
        assert isinstance(result, This)
        assert isinstance(result.interval, DayOfWeek)
        assert result.interval.type == DayOfWeekType.SATURDAY


class TestRobustness:
    """Tests for robustness and unusual inputs."""
    
    def test_whitespace_only_returns_none(self):
        """Test whitespace-only string returns None."""
        assert timeparser.parse("   ") is None
    
    def test_garbage_returns_none(self):
        """Test garbage input returns None."""
        assert timeparser.parse("???") is None
        assert timeparser.parse("abc123xyz") is None
    
    def test_case_insensitive_today(self):
        """Test TODAY, Today, TODAY all work."""
        assert isinstance(timeparser.parse("TODAY", languages=['en']), Today)
        assert isinstance(timeparser.parse("today", languages=['en']), Today)
        assert isinstance(timeparser.parse("ToDay", languages=['en']), Today)
    
    def test_case_insensitive_abbreviations(self):
        """Test RN, rn, Rn all work."""
        assert isinstance(timeparser.parse("RN", languages=['en']), Now)
        assert isinstance(timeparser.parse("rn", languages=['en']), Now)
        assert isinstance(timeparser.parse("ATM", languages=['en']), Now)
    
    def test_extra_whitespace(self):
        """Test extra whitespace is handled."""
        assert isinstance(timeparser.parse("  today  ", languages=['en']), Today)
        assert isinstance(timeparser.parse("3  days  ago", languages=['en']), Shift)
        assert isinstance(timeparser.parse("next   Monday", languages=['en']), Next)
    
    def test_typo_tolerance_tomorrow(self):
        """Test common typos of tomorrow work."""
        assert isinstance(timeparser.parse("tommorow", languages=['en']), Tomorrow)
        assert isinstance(timeparser.parse("tomorow", languages=['en']), Tomorrow)
    
    def test_punctuation_tolerance(self):
        """Test punctuation at end is handled."""
        assert isinstance(timeparser.parse("today.", languages=['en']), Today)
        assert isinstance(timeparser.parse("tomorrow!", languages=['en']), Tomorrow)
    
    def test_zero_days_ago_is_today(self):
        """Test '0 days ago' is Today."""
        result = timeparser.parse("0 days ago", languages=['en'])
        assert isinstance(result, Today)
    
    def test_fractional_days(self):
        """Test '1.5 days ago' works."""
        result = timeparser.parse("1.5 days ago", languages=['en'])
        assert isinstance(result, Shift)
    
    def test_large_numbers(self):
        """Test large numbers work."""
        result = timeparser.parse("100 years ago", languages=['en'])
        assert isinstance(result, Shift)
        result = timeparser.parse("1000 days ago", languages=['en'])
        assert isinstance(result, Shift)


class TestPatternEvaluation:
    """Tests for evaluating new pattern variants."""
    
    @pytest.fixture
    def anchor(self):
        """Provide a fixed anchor date for testing."""
        return datetime(2023, 10, 15, 12, 0, 0)
    
    def test_evaluate_day_before_yesterday(self, anchor):
        """Test evaluating 'the day before yesterday'."""
        result = timeparser.parse("the day before yesterday", languages=['en'])
        start, end = result.evaluate(anchor)
        # 2 days before Oct 15 = Oct 13
        assert start == datetime(2023, 10, 13, 0, 0, 0)
        assert end == datetime(2023, 10, 13, 23, 59, 59)
    
    def test_evaluate_day_after_tomorrow(self, anchor):
        """Test evaluating 'day after tomorrow'."""
        result = timeparser.parse("day after tomorrow", languages=['en'])
        start, end = result.evaluate(anchor)
        # 2 days after Oct 15 = Oct 17
        assert start == datetime(2023, 10, 17, 0, 0, 0)
        assert end == datetime(2023, 10, 17, 23, 59, 59)
    
    def test_evaluate_now_variants_same_as_now(self, anchor):
        """Test that all NOW variants evaluate identically."""
        now_result = timeparser.parse("now", languages=['en'])
        now_start, now_end = now_result.evaluate(anchor)
        
        for variant in ["right now", "just now", "rn", "atm", "currently"]:
            result = timeparser.parse(variant, languages=['en'])
            start, end = result.evaluate(anchor)
            assert start == now_start, f"'{variant}' start mismatch"
            assert end == now_end, f"'{variant}' end mismatch"
    
    def test_evaluate_today_variants_same_as_today(self, anchor):
        """Test that all TODAY variants evaluate identically."""
        today_result = timeparser.parse("today", languages=['en'])
        today_start, today_end = today_result.evaluate(anchor)
        
        for variant in ["2day", "tdy", "this day", "on this day"]:
            result = timeparser.parse(variant, languages=['en'])
            start, end = result.evaluate(anchor)
            assert start == today_start, f"'{variant}' start mismatch"
            assert end == today_end, f"'{variant}' end mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
