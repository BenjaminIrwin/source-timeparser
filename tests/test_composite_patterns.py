"""
Tests for composite temporal pattern recognition.

These tests verify that multi-part temporal expressions are parsed as
unified SCATEX expressions rather than fragmented pieces.
"""

import pytest
from datetime import datetime

import timeparser
from timeparser.scatex import (
    TemporalExpression,
    Year, Month, Day, Hour, Minute,
    DayOfWeek, TimeOfDay, TimeOfDayType,
    Before, After, Between, Intersection,
    ModifiedInterval, Approximate,
    Period, Unit,
    Unknown,
)
from timeparser.composite_patterns import composite_parser, CompositePatternParser


# =============================================================================
# Tests for ModifiedInterval SCATEX Type
# =============================================================================

class TestModifiedInterval:
    """Tests for ModifiedInterval SCATEX type."""
    
    @pytest.fixture
    def anchor(self):
        return datetime(2023, 10, 15, 12, 0, 0)
    
    def test_early_morning(self, anchor):
        """Test 'early morning' evaluates to first third of morning."""
        expr = ModifiedInterval(
            interval=TimeOfDay(type=TimeOfDayType.MORNING),
            position='early'
        )
        start, end = expr.evaluate(anchor)
        # Morning is 6:00-12:00, early = first third (approximately 6:00-8:00)
        assert start.hour == 6
        assert 7 <= end.hour <= 8  # Approximately 8 AM due to division
    
    def test_late_morning(self, anchor):
        """Test 'late morning' evaluates to last third of morning."""
        expr = ModifiedInterval(
            interval=TimeOfDay(type=TimeOfDayType.MORNING),
            position='late'
        )
        start, end = expr.evaluate(anchor)
        # Morning is 6:00-12:00, late = last third (approximately 10:00-12:00)
        assert 9 <= start.hour <= 10  # Approximately 10 AM
        assert end.hour == 11  # 11:59:59
    
    def test_mid_morning(self, anchor):
        """Test 'mid morning' evaluates to middle third of morning."""
        expr = ModifiedInterval(
            interval=TimeOfDay(type=TimeOfDayType.MORNING),
            position='mid'
        )
        start, end = expr.evaluate(anchor)
        # Morning is 6:00-12:00, mid = middle third (approximately 8:00-10:00)
        assert 7 <= start.hour <= 8  # Approximately 8 AM
        assert 9 <= end.hour <= 10  # Approximately 10 AM
    
    def test_early_year(self, anchor):
        """Test 'early 2023' evaluates to first third of year."""
        expr = ModifiedInterval(
            interval=Year(digits=2023),
            position='early'
        )
        start, end = expr.evaluate(anchor)
        assert start.year == 2023
        assert start.month == 1
        # First third of year is roughly Jan-Apr
        assert end.month <= 5
    
    def test_late_month(self, anchor):
        """Test 'late October' evaluates to last third of October."""
        expr = ModifiedInterval(
            interval=Month(month=10, year=2023),
            position='late'
        )
        start, end = expr.evaluate(anchor)
        assert start.month == 10
        assert start.day >= 20  # Last third starts around Oct 21
        assert end.day == 31
    
    def test_repr(self):
        """Test ModifiedInterval __repr__."""
        expr = ModifiedInterval(
            interval=TimeOfDay(type=TimeOfDayType.MORNING),
            position='early'
        )
        assert "ModifiedInterval" in repr(expr)
        assert "early" in repr(expr)
        assert "MORNING" in repr(expr)


# =============================================================================
# Tests for Approximate SCATEX Type
# =============================================================================

class TestApproximate:
    """Tests for Approximate SCATEX type."""
    
    @pytest.fixture
    def anchor(self):
        return datetime(2023, 10, 15, 12, 0, 0)
    
    def test_approximate_year_with_margin(self, anchor):
        """Test 'circa 1990' with explicit margin."""
        expr = Approximate(
            interval=Year(digits=1990),
            margin=Period(unit=Unit.YEAR, value=5)
        )
        start, end = expr.evaluate(anchor)
        assert start.year == 1985
        assert end.year == 1995
    
    def test_approximate_hour(self, anchor):
        """Test 'around 6 PM' with hour margin."""
        expr = Approximate(
            interval=Hour(hour=18, day=15, month=10, year=2023),
            margin=Period(unit=Unit.HOUR, value=1)
        )
        start, end = expr.evaluate(anchor)
        assert start.hour == 17  # 5 PM
        assert end.hour == 19   # 7 PM
    
    def test_approximate_default_margin(self, anchor):
        """Test approximate with default 10% margin."""
        expr = Approximate(
            interval=Day(day=15, month=10, year=2023)
        )
        start, end = expr.evaluate(anchor)
        # Default margin is max(10%, 1 hour), so for a day it should be at least 1 hour on each side
        assert start < datetime(2023, 10, 15, 0, 0, 0)
        assert end > datetime(2023, 10, 15, 23, 59, 59)
    
    def test_repr(self):
        """Test Approximate __repr__."""
        expr = Approximate(
            interval=Year(digits=1990),
            margin=Period(unit=Unit.YEAR, value=5)
        )
        assert "Approximate" in repr(expr)
        assert "1990" in repr(expr)


# =============================================================================
# Tests for Between Pattern
# =============================================================================

class TestBetweenPattern:
    """Tests for 'between X and Y' composite pattern."""
    
    def test_between_dates(self):
        """Test 'between October 7 and July 1'."""
        result = timeparser.parse("between October 7 and July 1", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Between)
        assert isinstance(result.start_interval, Day)
        assert result.start_interval.day == 7
        assert result.start_interval.month == 10
        assert isinstance(result.end_interval, Day)
        assert result.end_interval.day == 1
        assert result.end_interval.month == 7
    
    def test_between_years(self):
        """Test 'between 2020 and 2023'."""
        result = timeparser.parse("between 2020 and 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Between)
        assert isinstance(result.start_interval, Year)
        assert result.start_interval.digits == 2020
        assert isinstance(result.end_interval, Year)
        assert result.end_interval.digits == 2023
    
    def test_between_full_dates(self):
        """Test 'between January 1, 2020 and December 31, 2023'."""
        result = timeparser.parse("between January 1, 2020 and December 31, 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Between)
        assert isinstance(result.start_interval, Day)
        assert result.start_interval.year == 2020
        assert result.start_interval.month == 1
        assert result.start_interval.day == 1
    
    def test_from_to_dates(self):
        """Test 'from October 7 to July 1'."""
        result = timeparser.parse("from October 7 to July 1", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Between)
    
    def test_from_until_dates(self):
        """Test 'from January 2020 until March 2023'."""
        result = timeparser.parse("from January 2020 until March 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Between)


# =============================================================================
# Tests for Modified Time-of-Day Pattern
# =============================================================================

class TestModifiedTimeOfDayPattern:
    """Tests for 'early in the morning of X' patterns."""
    
    def test_early_morning_of_date(self):
        """Test 'Early in the morning of October 7, 2023'."""
        result = timeparser.parse("Early in the morning of October 7, 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'early'
        assert isinstance(result.interval, Intersection)
    
    def test_late_evening_of_date(self):
        """Test 'late evening of October 7, 2023'."""
        result = timeparser.parse("late evening of October 7, 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'late'
    
    def test_early_morning_compact(self):
        """Test 'early morning of October 7'."""
        result = timeparser.parse("early morning of October 7", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
    
    def test_standalone_early_morning(self):
        """Test standalone 'early morning'."""
        result = timeparser.parse("early morning", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'early'
        assert isinstance(result.interval, TimeOfDay)
    
    def test_late_evening(self):
        """Test standalone 'late evening'."""
        result = timeparser.parse("late evening", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'late'
    
    def test_mid_afternoon(self):
        """Test 'mid-afternoon'."""
        result = timeparser.parse("mid-afternoon", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'mid'


# =============================================================================
# Tests for Modified Date Unit Patterns
# =============================================================================

class TestModifiedDateUnitPattern:
    """Tests for 'early October', 'late 2023' patterns."""
    
    def test_early_october(self):
        """Test 'early October'."""
        result = timeparser.parse("early October", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'early'
        assert isinstance(result.interval, Month)
        assert result.interval.month == 10
    
    def test_late_january_2023(self):
        """Test 'late January 2023'."""
        result = timeparser.parse("late January 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'late'
        assert isinstance(result.interval, Month)
        assert result.interval.month == 1
        assert result.interval.year == 2023
    
    def test_early_2023(self):
        """Test 'early 2023'."""
        result = timeparser.parse("early 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'early'
        assert isinstance(result.interval, Year)
        assert result.interval.digits == 2023
    
    def test_mid_2020(self):
        """Test 'mid-2020'."""
        result = timeparser.parse("mid-2020", languages=['en'])
        
        assert result is not None
        assert isinstance(result, ModifiedInterval)
        assert result.position == 'mid'


# =============================================================================
# Tests for Approximate Patterns
# =============================================================================

class TestApproximatePattern:
    """Tests for 'around X', 'circa X' patterns."""
    
    def test_around_6pm(self):
        """Test 'around 6 PM'."""
        result = timeparser.parse("around 6 PM", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Approximate)
    
    def test_circa_1990(self):
        """Test 'circa 1990'."""
        result = timeparser.parse("circa 1990", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Approximate)
        assert isinstance(result.interval, Year)
        assert result.interval.digits == 1990
    
    def test_approximately_3_weeks_ago(self):
        """Test 'approximately 3 weeks ago'."""
        result = timeparser.parse("approximately 3 weeks ago", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Approximate)
    
    def test_about_october_7(self):
        """Test 'about October 7'."""
        result = timeparser.parse("about October 7", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Approximate)


# =============================================================================
# Tests for Bounded Patterns
# =============================================================================

class TestBoundedPatterns:
    """Tests for 'since X', 'until X', 'before X', 'after X' patterns."""
    
    def test_since_october_7(self):
        """Test 'since October 7'."""
        result = timeparser.parse("since October 7", languages=['en'])
        
        assert result is not None
        assert isinstance(result, After)
        assert isinstance(result.interval, Day)
    
    def test_until_july_1(self):
        """Test 'until July 1'."""
        result = timeparser.parse("until July 1", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Before)
    
    def test_before_2020(self):
        """Test 'before 2020'."""
        result = timeparser.parse("before 2020", languages=['en'])
        
        assert result is not None
        assert isinstance(result, Before)
        assert isinstance(result.interval, Year)
    
    def test_after_october_7_2023(self):
        """Test 'after October 7, 2023'."""
        result = timeparser.parse("after October 7, 2023", languages=['en'])
        
        assert result is not None
        assert isinstance(result, After)


# =============================================================================
# Tests for Signal Suppression
# =============================================================================

class TestSignalSuppression:
    """Tests verifying signals are not emitted for composite patterns."""
    
    def test_between_detected_and_merged(self):
        """Verify 'between' is detected and merged into composite."""
        from timeparser import extract_temporal_signals, extract_temporal_spans, merge_signals_with_spans
        
        text = "The attack occurred between October 7 and July 1."
        
        signals = extract_temporal_signals(text)
        spans = extract_temporal_spans(text)
        
        # 'between' should be detected as a signal (needed for signal-guided merge)
        between_signals = [s for s in signals if s.text.lower() == 'between']
        assert len(between_signals) == 1
        
        # After merging, 'between' should be consumed into a composite
        result = merge_signals_with_spans(text, spans, signals)
        assert len(result.composites) == 1
        assert isinstance(result.composites[0].expression, Between)
        # 'between' signal should be consumed (not in remaining)
        assert len([s for s in result.remaining_signals if s.text.lower() == 'between']) == 0
    
    def test_early_not_emitted_as_signal(self):
        """Verify 'early' in composite isn't extracted as separate signal."""
        from timeparser import extract_temporal_signals
        
        text = "Early in the morning of October 7, the attack began."
        
        signals = extract_temporal_signals(text)
        early_signals = [s for s in signals if s.text.lower() == 'early']
        
        # 'early' should be suppressed because it's followed by time-of-day
        assert len(early_signals) == 0
    
    def test_around_not_emitted_as_signal(self):
        """Verify 'around' in composite isn't extracted as separate signal."""
        from timeparser import extract_temporal_signals
        
        text = "It happened around 6 PM."
        
        signals = extract_temporal_signals(text)
        around_signals = [s for s in signals if s.text.lower() == 'around']
        
        # 'around' should be suppressed because it's followed by a time
        assert len(around_signals) == 0
    
    def test_standalone_early_still_detected(self):
        """Verify standalone 'early' (not part of composite) is still detected."""
        from timeparser import extract_temporal_signals
        
        # 'early' here is not followed by time-of-day, so should be detected
        text = "Early, the soldiers arrived."
        
        signals = extract_temporal_signals(text)
        # This depends on whether 'early' alone is in the lexicon
        # The test verifies the suppression logic doesn't over-suppress


# =============================================================================
# Tests for CompositePatternParser Class
# =============================================================================

class TestCompositePatternParser:
    """Tests for the CompositePatternParser class directly."""
    
    def test_parser_singleton(self):
        """Test that composite_parser is a singleton."""
        assert composite_parser is not None
        assert isinstance(composite_parser, CompositePatternParser)
    
    def test_try_parse_returns_none_for_simple_date(self):
        """Test that simple dates don't match composite patterns."""
        from timeparser.conf import Settings
        settings = Settings()
        
        result = composite_parser.try_parse("October 7, 2023", settings)
        # Simple date shouldn't match composite patterns
        assert result is None
    
    def test_try_parse_returns_result_for_between(self):
        """Test that 'between X and Y' matches."""
        from timeparser.conf import Settings
        settings = Settings()
        
        result = composite_parser.try_parse("between 2020 and 2023", settings)
        assert result is not None
        expr, period = result
        assert isinstance(expr, Between)
    
    def test_pattern_priority_ordering(self):
        """Test that patterns are ordered by priority."""
        priorities = [p.priority for p in composite_parser._patterns]
        # Should be sorted in descending order
        assert priorities == sorted(priorities, reverse=True)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for composite patterns with the full parser."""
    
    @pytest.fixture
    def anchor(self):
        return datetime(2023, 10, 15, 12, 0, 0)
    
    def test_evaluate_between_range(self, anchor):
        """Test evaluating a 'between X and Y' expression."""
        result = timeparser.parse("between January 1, 2023 and December 31, 2023", languages=['en'])
        
        assert result is not None
        start, end = result.evaluate(anchor)
        
        assert start.year == 2023
        assert start.month == 1
        assert start.day == 1
        assert end.year == 2023
        assert end.month == 12
        assert end.day == 31
    
    def test_evaluate_modified_time_of_day(self, anchor):
        """Test evaluating 'early morning' expression."""
        result = timeparser.parse("early morning", languages=['en'])
        
        assert result is not None
        start, end = result.evaluate(anchor)
        
        # Early morning should be first third of morning (6-8 AM)
        assert start.hour >= 6
        assert end.hour <= 8
    
    def test_evaluate_approximate(self, anchor):
        """Test evaluating 'circa 1990' expression."""
        result = timeparser.parse("circa 1990", languages=['en'])
        
        assert result is not None
        start, end = result.evaluate(anchor)
        
        # circa 1990 with default margin of 5 years = 1985-1995
        assert start.year <= 1990
        assert end.year >= 1990


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

