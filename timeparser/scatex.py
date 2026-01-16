"""
SCATEX (SCATE eXecutable) - Executable Temporal Expression Library

This module extends SCATE (Semantically Compositional Annotation of Temporal Expressions)
into an executable Python implementation. Based on the framework described in 
Su et al. "A Semantic Parsing Framework for End-to-End Time Normalization" (2507.06450v1).

Key extensions beyond SCATE:
- Executable operators: Each class implements .evaluate(anchor) -> (start, end) intervals
- Partial date support: Components can be None/unknown (e.g., Day(day=7) without month/year)
- Time-of-day intersection: Smart combining of day expressions with TimeOfDay patterns
- Safe code evaluation: evaluate_scatex_code() for executing SCATEX strings
- Convenience expressions: Today, Yesterday, Tomorrow, Decade, Century, Quarter, Unknown

Each operator implements .evaluate(anchor: datetime) -> (start: datetime, end: datetime)
where anchor is the reference datetime used to resolve relative temporal expressions.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Tuple, List, Union as UnionType
import calendar


# =============================================================================
# Enums for SCATEX types
# =============================================================================

class Unit(Enum):
    """Temporal units for intervals and shifts."""
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"
    YEAR = "YEAR"
    DECADE = "DECADE"
    CENTURY = "CENTURY"


class DayOfWeekType(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class MonthOfYearType(Enum):
    """Months of the year."""
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


class TimeOfDayType(Enum):
    """Common time-of-day references."""
    DAWN = (5, 6)       # 5:00 - 6:00
    MORNING = (6, 12)   # 6:00 - 12:00
    NOON = (12, 13)     # 12:00 - 13:00
    AFTERNOON = (12, 18)  # 12:00 - 18:00
    EVENING = (18, 21)  # 18:00 - 21:00
    NIGHT = (21, 24)    # 21:00 - 24:00
    MIDNIGHT = (0, 1)   # 0:00 - 1:00


class Direction(Enum):
    """Direction for shift operations."""
    BEFORE = "BEFORE"
    AFTER = "AFTER"


# =============================================================================
# Base Classes
# =============================================================================

class TemporalExpression(ABC):
    """Base class for all temporal expressions in SCATEX."""
    
    @abstractmethod
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Evaluate this temporal expression relative to the anchor datetime.
        
        Args:
            anchor: Reference datetime used to resolve relative expressions
            
        Returns:
            Tuple of (start, end) datetimes. Either may be None for unbounded intervals.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RepeatingExpression(ABC):
    """Base class for repeating temporal patterns (e.g., "every Monday")."""
    
    @abstractmethod
    def get_instance(self, anchor: datetime, offset: int = 0) -> Tuple[datetime, datetime]:
        """
        Get a specific instance of this repeating pattern.
        
        Args:
            anchor: Reference datetime
            offset: 0 for current/containing, -1 for previous, +1 for next, etc.
            
        Returns:
            Tuple of (start, end) for that instance
        """
        pass


# =============================================================================
# Core Interval Types (Specific Points/Periods)
# =============================================================================

@dataclass
class Year(TemporalExpression):
    """A specific year (e.g., Year(2023))."""
    digits: int
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        start = datetime(self.digits, 1, 1, 0, 0, 0)
        end = datetime(self.digits, 12, 31, 23, 59, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        return f"Year(digits={self.digits})"


@dataclass
class Month(TemporalExpression):
    """
    A specific month. Year is optional for partial dates.
    
    Parameters are ordered smallest→largest: month, year.
    
    Examples:
        Month(month=10, year=2023)  # Full: October 2023
        Month(month=10)              # Partial: October (year unknown)
    """
    month: int
    year: Optional[int] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        if self.year is None:
            # Cannot evaluate without year
            return (None, None)
        start = datetime(self.year, self.month, 1, 0, 0, 0)
        last_day = calendar.monthrange(self.year, self.month)[1]
        end = datetime(self.year, self.month, last_day, 23, 59, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        parts = [f"month={self.month}"]
        if self.year is not None:
            parts.append(f"year={self.year}")
        return f"Month({', '.join(parts)})"


@dataclass
class Day(TemporalExpression):
    """
    A specific day. Month and year are optional for partial dates.
    
    Parameters are ordered smallest→largest: day, month, year.
    
    Examples:
        Day(day=7, month=10, year=2023)  # Full date: October 7, 2023
        Day(day=7, month=10)              # Partial: October 7 (year unknown)
        Day(day=7)                        # Partial: the 7th (month, year unknown)
    """
    day: int
    month: Optional[int] = None
    year: Optional[int] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        if self.year is None or self.month is None:
            # Cannot evaluate without full date context
            return (None, None)
        start = datetime(self.year, self.month, self.day, 0, 0, 0)
        end = datetime(self.year, self.month, self.day, 23, 59, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        parts = [f"day={self.day}"]
        if self.month is not None:
            parts.append(f"month={self.month}")
        if self.year is not None:
            parts.append(f"year={self.year}")
        return f"Day({', '.join(parts)})"


@dataclass
class Hour(TemporalExpression):
    """
    A specific hour. Day, month, year are optional for partial times.
    
    Parameters are ordered smallest→largest: hour, day, month, year.
    
    Examples:
        Hour(hour=14, day=7, month=10, year=2023)  # Full: 2 PM on Oct 7, 2023
        Hour(hour=14)                               # Partial: 2 PM (date unknown)
    """
    hour: int
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        if self.year is None or self.month is None or self.day is None:
            # Cannot evaluate without full date context
            return (None, None)
        start = datetime(self.year, self.month, self.day, self.hour, 0, 0)
        end = datetime(self.year, self.month, self.day, self.hour, 59, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        parts = [f"hour={self.hour}"]
        if self.day is not None:
            parts.append(f"day={self.day}")
        if self.month is not None:
            parts.append(f"month={self.month}")
        if self.year is not None:
            parts.append(f"year={self.year}")
        return f"Hour({', '.join(parts)})"


@dataclass
class Minute(TemporalExpression):
    """
    A specific minute. Hour, day, month, year are optional for partial times.
    
    Parameters are ordered smallest→largest: minute, hour, day, month, year.
    
    Examples:
        Minute(minute=30, hour=14, day=7, month=10, year=2023)  # Full
        Minute(minute=30, hour=14)                               # Partial: 2:30 PM
        Minute(minute=30)                                        # Partial: :30 (context unknown)
    """
    minute: int
    hour: Optional[int] = None
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        if self.year is None or self.month is None or self.day is None or self.hour is None:
            # Cannot evaluate without full datetime context
            return (None, None)
        start = datetime(self.year, self.month, self.day, self.hour, self.minute, 0)
        end = datetime(self.year, self.month, self.day, self.hour, self.minute, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        parts = [f"minute={self.minute}"]
        if self.hour is not None:
            parts.append(f"hour={self.hour}")
        if self.day is not None:
            parts.append(f"day={self.day}")
        if self.month is not None:
            parts.append(f"month={self.month}")
        if self.year is not None:
            parts.append(f"year={self.year}")
        return f"Minute({', '.join(parts)})"


@dataclass
class Second(TemporalExpression):
    """
    A specific second (instant). All larger units are optional for partial times.
    
    Parameters are ordered smallest→largest: second, minute, hour, day, month, year.
    
    Examples:
        Second(second=0, minute=30, hour=14, day=7, month=10, year=2023)  # Full
        Second(second=0)  # Partial: :00 seconds
    """
    second: int
    minute: Optional[int] = None
    hour: Optional[int] = None
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        if (self.year is None or self.month is None or self.day is None or 
            self.hour is None or self.minute is None):
            # Cannot evaluate without full datetime context
            return (None, None)
        dt = datetime(self.year, self.month, self.day, self.hour, self.minute, self.second)
        return (dt, dt)
    
    def __repr__(self) -> str:
        parts = [f"second={self.second}"]
        if self.minute is not None:
            parts.append(f"minute={self.minute}")
        if self.hour is not None:
            parts.append(f"hour={self.hour}")
        if self.day is not None:
            parts.append(f"day={self.day}")
        if self.month is not None:
            parts.append(f"month={self.month}")
        if self.year is not None:
            parts.append(f"year={self.year}")
        return f"Second({', '.join(parts)})"


@dataclass
class Instant(TemporalExpression):
    """A specific instant in time (convenience wrapper for datetime)."""
    dt: datetime
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        return (self.dt, self.dt)
    
    def __repr__(self) -> str:
        return f"Instant(dt={self.dt.isoformat()})"


@dataclass
class Interval(TemporalExpression):
    """An explicit interval with start and end."""
    start: datetime
    end: datetime
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        return (self.start, self.end)
    
    def __repr__(self) -> str:
        return f"Interval(start={self.start.isoformat()}, end={self.end.isoformat()})"


# =============================================================================
# Repeating Intervals (Patterns)
# =============================================================================

@dataclass
class Repeating(RepeatingExpression, TemporalExpression):
    """
    A repeating interval pattern (e.g., "days", "months", "Mondays").
    
    Used to represent patterns like:
    - Repeating(unit=DAY, range=MONTH, value=15) -> "the 15th of each month"
    - Repeating(unit=MONTH, range=YEAR, value=4) -> "April of each year"
    """
    unit: Unit
    range: Optional[Unit] = None  # The containing unit (e.g., MONTH for day-of-month)
    value: Optional[int] = None   # Specific value within range (e.g., 15 for 15th day)
    
    def get_instance(self, anchor: datetime, offset: int = 0) -> Tuple[datetime, datetime]:
        """Get instance of this repeating pattern at offset from anchor."""
        if self.unit == Unit.YEAR:
            year = anchor.year + offset
            if self.value:
                year = self.value + offset
            return Year(digits=year).evaluate(anchor)
        
        elif self.unit == Unit.MONTH:
            if self.value and self.range == Unit.YEAR:
                # Specific month (e.g., April)
                year = anchor.year
                month = self.value
                if offset != 0:
                    year += offset
                return Month(year=year, month=month).evaluate(anchor)
            else:
                # Generic month offset
                year = anchor.year
                month = anchor.month + offset
                while month > 12:
                    month -= 12
                    year += 1
                while month < 1:
                    month += 12
                    year -= 1
                return Month(year=year, month=month).evaluate(anchor)
        
        elif self.unit == Unit.WEEK:
            # Weeks from start of year or from current
            days_offset = offset * 7
            target = anchor + timedelta(days=days_offset)
            # Get start of week (Monday)
            start = target - timedelta(days=target.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return (start, end)
        
        elif self.unit == Unit.DAY:
            if self.value and self.range == Unit.MONTH:
                # Specific day of month
                year = anchor.year
                month = anchor.month + offset
                while month > 12:
                    month -= 12
                    year += 1
                while month < 1:
                    month += 12
                    year -= 1
                day = min(self.value, calendar.monthrange(year, month)[1])
                return Day(year=year, month=month, day=day).evaluate(anchor)
            elif self.value and self.range == Unit.WEEK:
                # Specific day of week (0=Monday, 6=Sunday)
                current_weekday = anchor.weekday()
                target_weekday = self.value
                days_diff = target_weekday - current_weekday + (offset * 7)
                target = anchor + timedelta(days=days_diff)
                return Day(year=target.year, month=target.month, day=target.day).evaluate(anchor)
            else:
                # Generic day offset
                target = anchor + timedelta(days=offset)
                return Day(year=target.year, month=target.month, day=target.day).evaluate(anchor)
        
        elif self.unit == Unit.HOUR:
            target = anchor + timedelta(hours=offset)
            if self.value:
                target = target.replace(hour=self.value)
            return Hour(year=target.year, month=target.month, day=target.day, hour=target.hour).evaluate(anchor)
        
        # Default fallback
        return (anchor, anchor)
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        """Evaluate as the current instance containing anchor."""
        return self.get_instance(anchor, 0)
    
    def __repr__(self) -> str:
        parts = [f"unit={self.unit.value}"]
        if self.range:
            parts.append(f"range={self.range.value}")
        if self.value is not None:
            parts.append(f"value={self.value}")
        return f"Repeating({', '.join(parts)})"


@dataclass  
class DayOfWeek(RepeatingExpression, TemporalExpression):
    """A specific day of the week (e.g., Monday, Tuesday)."""
    type: DayOfWeekType
    
    def get_instance(self, anchor: datetime, offset: int = 0) -> Tuple[datetime, datetime]:
        """Get the nth occurrence of this weekday relative to anchor."""
        current_weekday = anchor.weekday()
        target_weekday = self.type.value
        
        # Calculate days to target weekday
        days_diff = target_weekday - current_weekday
        if offset == 0:
            # Current week's instance
            pass
        elif offset < 0:
            # Previous instances
            if days_diff >= 0:
                days_diff -= 7
            days_diff += (offset + 1) * 7
        else:
            # Future instances
            if days_diff <= 0:
                days_diff += 7
            days_diff += (offset - 1) * 7
        
        target = anchor + timedelta(days=days_diff)
        return Day(year=target.year, month=target.month, day=target.day).evaluate(anchor)
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        """Return the current week's instance of this day."""
        return self.get_instance(anchor, 0)
    
    def __repr__(self) -> str:
        return f"DayOfWeek(type={self.type.name})"


@dataclass
class MonthOfYear(RepeatingExpression, TemporalExpression):
    """A specific month of the year (e.g., January, April)."""
    type: MonthOfYearType
    
    def get_instance(self, anchor: datetime, offset: int = 0) -> Tuple[datetime, datetime]:
        """Get the nth occurrence of this month relative to anchor."""
        year = anchor.year + offset
        if offset == 0 and anchor.month > self.type.value:
            # This year's instance has passed
            pass
        return Month(year=year, month=self.type.value).evaluate(anchor)
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        """Return this year's instance of this month."""
        return self.get_instance(anchor, 0)
    
    def __repr__(self) -> str:
        return f"MonthOfYear(type={self.type.name})"


@dataclass
class TimeOfDay(RepeatingExpression, TemporalExpression):
    """A time of day period (e.g., morning, evening)."""
    type: TimeOfDayType
    
    def get_instance(self, anchor: datetime, offset: int = 0) -> Tuple[datetime, datetime]:
        """Get the nth occurrence of this time period."""
        target_date = anchor + timedelta(days=offset)
        start_hour, end_hour = self.type.value
        
        start = target_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        if end_hour == 24:
            end = target_date.replace(hour=23, minute=59, second=59, microsecond=0)
        else:
            end = target_date.replace(hour=end_hour - 1, minute=59, second=59, microsecond=0)
        
        return (start, end)
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        """Return today's instance of this time period."""
        return self.get_instance(anchor, 0)
    
    def __repr__(self) -> str:
        return f"TimeOfDay(type={self.type.name})"


@dataclass
class RepeatingIntersection(RepeatingExpression, TemporalExpression):
    """
    Intersection of multiple repeating patterns.
    
    Example: "April 30" = RepeatingIntersection(shifts=[
        Repeating(unit=MONTH, range=YEAR, value=4),   # April
        Repeating(unit=DAY, range=MONTH, value=30)     # 30th
    ])
    """
    shifts: List[Repeating]
    
    def get_instance(self, anchor: datetime, offset: int = 0) -> Tuple[datetime, datetime]:
        """Resolve the intersection to a concrete interval."""
        # Start with the DCT and apply each constraint
        year = anchor.year + offset
        month = anchor.month
        day = anchor.day
        hour = 0
        minute = 0
        
        for shift in self.shifts:
            if shift.unit == Unit.MONTH and shift.range == Unit.YEAR and shift.value:
                month = shift.value
            elif shift.unit == Unit.DAY and shift.range == Unit.MONTH and shift.value:
                day = shift.value
            elif shift.unit == Unit.DAY and shift.range == Unit.WEEK and shift.value is not None:
                # Day of week - need to find the matching day
                target = datetime(year, month, 1)
                while target.weekday() != shift.value:
                    target += timedelta(days=1)
                day = target.day
            elif shift.unit == Unit.HOUR and shift.value:
                hour = shift.value
            elif shift.unit == Unit.MINUTE and shift.value:
                minute = shift.value
        
        # Validate day for month
        last_day = calendar.monthrange(year, month)[1]
        day = min(day, last_day)
        
        try:
            start = datetime(year, month, day, hour, minute, 0)
            end = datetime(year, month, day, 23, 59, 59)
            return (start, end)
        except ValueError:
            # Invalid date, return None
            return (None, None)
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        return self.get_instance(anchor, 0)
    
    def __repr__(self) -> str:
        shifts_repr = ", ".join(repr(s) for s in self.shifts)
        return f"RepeatingIntersection(shifts=[{shifts_repr}])"


# =============================================================================
# DCT-Relative Operators (This, Last, Next)
# =============================================================================

@dataclass
class This(TemporalExpression):
    """
    The current instance of a repeating pattern relative to DCT.
    
    Examples:
    - This(interval=Repeating(unit=WEEK)) -> "this week"
    - This(interval=Year(digits=1998), shift=RepeatingIntersection(...)) -> specific date in 1998
    """
    interval: UnionType[TemporalExpression, RepeatingExpression]
    shift: Optional[UnionType[RepeatingExpression, TemporalExpression]] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        if self.shift:
            # First evaluate the base interval, then apply the shift
            base_start, base_end = self.interval.evaluate(anchor)
            if base_start is None:
                return (None, None)
            # Use base_start as the reference for the shift
            return self.shift.evaluate(base_start)
        else:
            if isinstance(self.interval, RepeatingExpression):
                return self.interval.get_instance(anchor, 0)
            else:
                return self.interval.evaluate(anchor)
    
    def __repr__(self) -> str:
        if self.shift:
            return f"This(interval={self.interval!r}, shift={self.shift!r})"
        return f"This(interval={self.interval!r})"


@dataclass
class Last(TemporalExpression):
    """
    The previous instance of a repeating pattern relative to DCT.
    
    Examples:
    - Last(interval=DayOfWeek(type=MONDAY)) -> "last Monday"
    - Last(interval=Repeating(unit=WEEK)) -> "last week"
    """
    interval: UnionType[TemporalExpression, RepeatingExpression]
    shift: Optional[RepeatingExpression] = None
    count: int = 1  # For "last 3 weeks" etc.
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        if isinstance(self.interval, RepeatingExpression):
            return self.interval.get_instance(anchor, -self.count)
        else:
            # For non-repeating, shift backwards
            start, end = self.interval.evaluate(anchor)
            if start is None:
                return (None, None)
            # Apply shift if present
            if self.shift:
                return self.shift.get_instance(start, -self.count)
            return (start, end)
    
    def __repr__(self) -> str:
        parts = [f"interval={self.interval!r}"]
        if self.shift:
            parts.append(f"shift={self.shift!r}")
        if self.count != 1:
            parts.append(f"count={self.count}")
        return f"Last({', '.join(parts)})"


@dataclass
class Next(TemporalExpression):
    """
    The next instance of a repeating pattern relative to DCT.
    
    Examples:
    - Next(interval=DayOfWeek(type=FRIDAY)) -> "next Friday"
    - Next(interval=MonthOfYear(type=JANUARY)) -> "next January"
    """
    interval: UnionType[TemporalExpression, RepeatingExpression]
    shift: Optional[RepeatingExpression] = None
    count: int = 1
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        if isinstance(self.interval, RepeatingExpression):
            return self.interval.get_instance(anchor, self.count)
        else:
            start, end = self.interval.evaluate(anchor)
            if start is None:
                return (None, None)
            if self.shift:
                return self.shift.get_instance(start, self.count)
            return (start, end)
    
    def __repr__(self) -> str:
        parts = [f"interval={self.interval!r}"]
        if self.shift:
            parts.append(f"shift={self.shift!r}")
        if self.count != 1:
            parts.append(f"count={self.count}")
        return f"Next({', '.join(parts)})"


# =============================================================================
# Relative Shift Operators
# =============================================================================

@dataclass
class Shift(TemporalExpression):
    """
    Shift an interval by a period.
    
    Examples:
    - Shift(interval=This(Repeating(unit=DAY)), period=Period(unit=DAY, value=3), direction=BEFORE) 
      -> "3 days ago"
    - Shift(interval=EventRef("surgery"), period=Period(unit=WEEK, value=3), direction=AFTER)
      -> "3 weeks post-operative"
    """
    interval: TemporalExpression
    period: 'Period'
    direction: Direction = Direction.BEFORE
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        start, end = self.interval.evaluate(anchor)
        if start is None:
            return (None, None)
        
        delta = self.period.to_timedelta()
        
        if self.direction == Direction.BEFORE:
            return (start - delta, end - delta if end else None)
        else:
            return (start + delta, end + delta if end else None)
    
    def __repr__(self) -> str:
        return f"Shift(interval={self.interval!r}, period={self.period!r}, direction={self.direction.value})"


@dataclass
class Period:
    """
    A duration/period of time (e.g., 3 days, 2 weeks).
    
    NOTE: Period is NOT a TemporalExpression because durations don't have
    anchor points - they represent a length of time, not an interval.
    Period is used as a component in Shift operations.
    """
    unit: Unit
    value: int = 1
    
    def to_timedelta(self) -> timedelta:
        """Convert this period to a timedelta."""
        if self.unit == Unit.SECOND:
            return timedelta(seconds=self.value)
        elif self.unit == Unit.MINUTE:
            return timedelta(minutes=self.value)
        elif self.unit == Unit.HOUR:
            return timedelta(hours=self.value)
        elif self.unit == Unit.DAY:
            return timedelta(days=self.value)
        elif self.unit == Unit.WEEK:
            return timedelta(weeks=self.value)
        elif self.unit == Unit.MONTH:
            return timedelta(days=self.value * 30)  # Approximation
        elif self.unit == Unit.QUARTER:
            return timedelta(days=self.value * 91)  # Approximation
        elif self.unit == Unit.YEAR:
            return timedelta(days=self.value * 365)  # Approximation
        elif self.unit == Unit.DECADE:
            return timedelta(days=self.value * 3650)
        elif self.unit == Unit.CENTURY:
            return timedelta(days=self.value * 36500)
        else:
            return timedelta(0)
    
    def __repr__(self) -> str:
        return f"Period(unit={self.unit.value}, value={self.value})"


@dataclass
class Before(TemporalExpression):
    """
    Time before a reference point.
    
    Example: Before(interval=Day(2023, 10, 7)) -> "before October 7, 2023"
    Returns an unbounded interval ending at the reference.
    """
    interval: TemporalExpression
    shift: Optional[Period] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], datetime]:
        start, end = self.interval.evaluate(anchor)
        ref_point = start if start else end
        
        if self.shift:
            ref_point = ref_point - self.shift.to_timedelta()
        
        return (None, ref_point)
    
    def __repr__(self) -> str:
        if self.shift:
            return f"Before(interval={self.interval!r}, shift={self.shift!r})"
        return f"Before(interval={self.interval!r})"


@dataclass
class After(TemporalExpression):
    """
    Time after a reference point.
    
    Example: After(interval=Day(2023, 10, 7)) -> "after October 7, 2023"
    Returns an unbounded interval starting at the reference.
    """
    interval: TemporalExpression
    shift: Optional[Period] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, Optional[datetime]]:
        start, end = self.interval.evaluate(anchor)
        ref_point = end if end else start
        
        if self.shift:
            ref_point = ref_point + self.shift.to_timedelta()
        
        return (ref_point, None)
    
    def __repr__(self) -> str:
        if self.shift:
            return f"After(interval={self.interval!r}, shift={self.shift!r})"
        return f"After(interval={self.interval!r})"


@dataclass
class Between(TemporalExpression):
    """
    Interval between two temporal expressions.
    
    Example: Between(start=Day(2023,1,1), end=Day(2023,12,31)) -> "during 2023"
    """
    start_interval: TemporalExpression
    end_interval: TemporalExpression
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        start, _ = self.start_interval.evaluate(anchor)
        _, end = self.end_interval.evaluate(anchor)
        return (start, end)
    
    def __repr__(self) -> str:
        return f"Between(start_interval={self.start_interval!r}, end_interval={self.end_interval!r})"


@dataclass
class ModifiedInterval(TemporalExpression):
    """
    A temporal interval with a position modifier (early, late, mid).
    
    The modifier divides the interval into thirds and returns the appropriate portion:
    - 'early': first third of the interval
    - 'mid': middle third of the interval
    - 'late': last third of the interval
    
    Examples:
        ModifiedInterval(interval=TimeOfDay(type=MORNING), position='early')
            -> "early morning" (6:00-8:00 if morning is 6:00-12:00)
        ModifiedInterval(interval=Month(month=10, year=2023), position='late')
            -> "late October 2023" (roughly Oct 21-31)
        ModifiedInterval(interval=Year(digits=2023), position='mid')
            -> "mid-2023" (roughly May-August)
    """
    interval: TemporalExpression
    position: str  # 'early', 'mid', 'late'
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        start, end = self.interval.evaluate(anchor)
        if start is None or end is None:
            return (None, None)
        
        # Calculate total duration
        duration = end - start
        third = duration / 3
        
        if self.position == 'early':
            # First third of interval
            return (start, start + third)
        elif self.position == 'late':
            # Last third of interval
            return (start + 2 * third, end)
        elif self.position == 'mid':
            # Middle third of interval
            return (start + third, start + 2 * third)
        else:
            # Unknown position, return full interval
            return (start, end)
    
    def __repr__(self) -> str:
        return f"ModifiedInterval(interval={self.interval!r}, position={self.position!r})"


@dataclass
class Approximate(TemporalExpression):
    """
    An approximate temporal reference with uncertainty margin.
    
    The margin extends the interval in both directions. If no margin is specified,
    a default margin of 10% of the interval duration is used.
    
    Examples:
        Approximate(interval=Hour(hour=18), margin=Period(unit=HOUR, value=1))
            -> "around 6 PM" (5 PM to 7 PM)
        Approximate(interval=Year(digits=1990), margin=Period(unit=YEAR, value=5))
            -> "circa 1990" (1985-1995)
        Approximate(interval=Day(day=7, month=10, year=2023))
            -> "around October 7, 2023" (with default 10% margin)
    """
    interval: TemporalExpression
    margin: Optional[Period] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        start, end = self.interval.evaluate(anchor)
        if start is None or end is None:
            return (None, None)
        
        if self.margin:
            # Use explicit margin
            delta = self.margin.to_timedelta()
            return (start - delta, end + delta)
        
        # Default: extend by 10% of interval duration on each side
        duration = end - start
        # Minimum margin of 1 hour for very short intervals
        margin = max(duration * 0.1, timedelta(hours=1))
        return (start - margin, end + margin)
    
    def __repr__(self) -> str:
        if self.margin:
            return f"Approximate(interval={self.interval!r}, margin={self.margin!r})"
        return f"Approximate(interval={self.interval!r})"


# =============================================================================
# Set Operations
# =============================================================================

@dataclass
class Union(TemporalExpression):
    """
    Union of multiple temporal expressions.
    
    Example: Union([DayOfWeek(MONDAY), DayOfWeek(FRIDAY)]) -> "Mondays and Fridays"
    
    For evaluation, returns the bounding interval containing all members.
    """
    intervals: List[TemporalExpression]
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        """Return the bounding interval of the union."""
        starts = []
        ends = []
        
        for interval in self.intervals:
            start, end = interval.evaluate(anchor)
            if start:
                starts.append(start)
            if end:
                ends.append(end)
        
        min_start = min(starts) if starts else None
        max_end = max(ends) if ends else None
        
        return (min_start, max_end)
    
    def __repr__(self) -> str:
        intervals_repr = ", ".join(repr(i) for i in self.intervals)
        return f"Union(intervals=[{intervals_repr}])"


@dataclass
class Intersection(TemporalExpression):
    """
    Intersection of multiple temporal expressions.
    
    Example: Intersection([TimeOfDay(MORNING), DayOfWeek(MONDAY)]) 
             -> "Monday mornings"
    
    Special handling for date + time-of-day combinations:
    - If one interval is a day-level expression and another is TimeOfDay,
      the TimeOfDay is applied to the day from the first expression.
    """
    intervals: List[TemporalExpression]
    
    def _is_day_expr(self, expr: TemporalExpression) -> bool:
        """Check if an expression evaluates to a day-level interval."""
        if isinstance(expr, (Today, Tomorrow, Yesterday, Day, DayOfWeek)):
            return True
        # Check for Next/Last/This wrapping day expressions
        if isinstance(expr, (Next, Last, This)):
            inner = expr.interval
            if isinstance(inner, (DayOfWeek, MonthOfYear)):
                return True
            # Only treat Repeating as day-level if it's actually day/week granularity
            if isinstance(inner, Repeating) and inner.unit in (Unit.DAY, Unit.WEEK):
                return True
        return False
    
    def _is_time_of_day_expr(self, expr: TemporalExpression) -> bool:
        """Check if an expression represents a time-of-day pattern."""
        if isinstance(expr, TimeOfDay):
            return True
        # Repeating with unit=HOUR and range=DAY represents a specific hour (e.g., "9 AM")
        if isinstance(expr, Repeating):
            if expr.unit == Unit.HOUR and expr.range == Unit.DAY:
                return True
        return False
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Return the intersection of all intervals."""
        if not self.intervals:
            return (None, None)
        
        # Separate day-level expressions from time-of-day expressions
        day_exprs = []
        time_exprs = []
        other_exprs = []
        
        for interval in self.intervals:
            if self._is_time_of_day_expr(interval):
                time_exprs.append(interval)
            elif self._is_day_expr(interval):
                day_exprs.append(interval)
            else:
                other_exprs.append(interval)
        
        # If we have day + time-of-day, combine them specially
        if day_exprs and time_exprs:
            # Get the day from the day expression
            day_start, day_end = day_exprs[0].evaluate(anchor)
            if day_start is None:
                return (None, None)
            
            # Apply time-of-day to that specific day
            result_start = day_start
            result_end = day_end
            
            for time_expr in time_exprs:
                if isinstance(time_expr, Repeating) and time_expr.unit == Unit.HOUR:
                    # Repeating hour pattern (e.g., "9 AM")
                    hour_val = time_expr.value if time_expr.value else 0
                    result_start = result_start.replace(hour=hour_val, minute=0, second=0)
                    result_end = result_end.replace(hour=hour_val, minute=59, second=59)
                elif isinstance(time_expr, TimeOfDay):
                    # Get time-of-day using the day's date as reference
                    time_start, time_end = time_expr.get_instance(day_start, 0)
                    
                    # Combine: take the time portion from time_expr
                    if time_start:
                        result_start = result_start.replace(
                            hour=time_start.hour, 
                            minute=time_start.minute, 
                            second=time_start.second
                        )
                    if time_end:
                        result_end = result_end.replace(
                            hour=time_end.hour, 
                            minute=time_end.minute, 
                            second=time_end.second
                        )
                else:
                    # Other time expressions - use get_instance if available
                    if hasattr(time_expr, 'get_instance'):
                        time_start, time_end = time_expr.get_instance(day_start, 0)
                        if time_start:
                            result_start = result_start.replace(
                                hour=time_start.hour, 
                                minute=time_start.minute, 
                                second=time_start.second
                            )
                        if time_end:
                            result_end = result_end.replace(
                                hour=time_end.hour, 
                                minute=time_end.minute, 
                                second=time_end.second
                            )
            
            # Intersect with any remaining day expressions
            for day_expr in day_exprs[1:]:
                start, end = day_expr.evaluate(anchor)
                if start and result_start:
                    result_start = max(start, result_start)
                if end and result_end:
                    result_end = min(end, result_end)
            
            # Intersect with other expressions
            for other_expr in other_exprs:
                start, end = other_expr.evaluate(anchor)
                if start and result_start:
                    result_start = max(start, result_start)
                elif start:
                    result_start = start
                if end and result_end:
                    result_end = min(end, result_end)
                elif end:
                    result_end = end
            
            if result_start and result_end and result_start > result_end:
                return (None, None)
            
            return (result_start, result_end)
        
        # Standard intersection logic for non-day+time combinations
        result_start, result_end = self.intervals[0].evaluate(anchor)
        
        for interval in self.intervals[1:]:
            start, end = interval.evaluate(anchor)
            
            # Intersect: take later start, earlier end
            if start and result_start:
                result_start = max(start, result_start)
            elif start:
                result_start = start
                
            if end and result_end:
                result_end = min(end, result_end)
            elif end:
                result_end = end
        
        # Check if intersection is valid (start <= end)
        if result_start and result_end and result_start > result_end:
            return (None, None)
        
        return (result_start, result_end)
    
    def __repr__(self) -> str:
        intervals_repr = ", ".join(repr(i) for i in self.intervals)
        return f"Intersection(intervals=[{intervals_repr}])"


# =============================================================================
# Special/Convenience Expressions
# =============================================================================

@dataclass
class Now(TemporalExpression):
    """The document creation time itself (instant)."""
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        return (anchor, anchor)
    
    def __repr__(self) -> str:
        return "Now()"


@dataclass
class Today(TemporalExpression):
    """Today (the day containing DCT)."""
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        start = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        end = anchor.replace(hour=23, minute=59, second=59, microsecond=0)
        return (start, end)
    
    def __repr__(self) -> str:
        return "Today()"


@dataclass
class Yesterday(TemporalExpression):
    """Yesterday."""
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        yesterday = anchor - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
        return (start, end)
    
    def __repr__(self) -> str:
        return "Yesterday()"


@dataclass
class Tomorrow(TemporalExpression):
    """Tomorrow."""
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        tomorrow = anchor + timedelta(days=1)
        start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        end = tomorrow.replace(hour=23, minute=59, second=59, microsecond=0)
        return (start, end)
    
    def __repr__(self) -> str:
        return "Tomorrow()"


@dataclass
class Decade(TemporalExpression):
    """A decade (e.g., the 1990s)."""
    start_year: int  # e.g., 1990 for "the 1990s"
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        start = datetime(self.start_year, 1, 1, 0, 0, 0)
        end = datetime(self.start_year + 9, 12, 31, 23, 59, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        return f"Decade(start_year={self.start_year})"


@dataclass
class Century(TemporalExpression):
    """A century (e.g., the 20th century = 1900-1999)."""
    number: int  # e.g., 20 for "20th century"
    
    def evaluate(self, anchor: datetime) -> Tuple[datetime, datetime]:
        start_year = (self.number - 1) * 100
        end_year = start_year + 99
        start = datetime(start_year, 1, 1, 0, 0, 0)
        end = datetime(end_year, 12, 31, 23, 59, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        return f"Century(number={self.number})"


@dataclass
class Quarter(TemporalExpression):
    """
    A fiscal quarter (Q1-Q4). Year is optional for partial dates.
    
    Parameters are ordered smallest→largest: quarter, year.
    
    Examples:
        Quarter(quarter=2, year=2023)  # Full: Q2 2023
        Quarter(quarter=2)              # Partial: Q2 (year unknown)
    """
    quarter: int  # 1-4
    year: Optional[int] = None
    
    def evaluate(self, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        if self.year is None:
            # Cannot evaluate without year
            return (None, None)
        start_month = (self.quarter - 1) * 3 + 1
        end_month = start_month + 2
        start = datetime(self.year, start_month, 1, 0, 0, 0)
        last_day = calendar.monthrange(self.year, end_month)[1]
        end = datetime(self.year, end_month, last_day, 23, 59, 59)
        return (start, end)
    
    def __repr__(self) -> str:
        parts = [f"quarter={self.quarter}"]
        if self.year is not None:
            parts.append(f"year={self.year}")
        return f"Quarter({', '.join(parts)})"


@dataclass
class Unknown(TemporalExpression):
    """Unknown or unresolvable temporal expression."""
    reason: str = ""
    
    def evaluate(self, anchor: datetime) -> Tuple[None, None]:
        return (None, None)
    
    def __repr__(self) -> str:
        if self.reason:
            return f"Unknown(reason={self.reason!r})"
        return "Unknown()"


# =============================================================================
# Execution Helper
# =============================================================================

def evaluate_scatex_code(code: str, anchor: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Safely evaluate SCATEX code and return the resulting interval.
    
    Args:
        code: SCATEX Python code string (e.g., "Year(digits=2023)")
        anchor: Reference datetime for resolving relative expressions
        
    Returns:
        Tuple of (start, end) datetimes
        
    Raises:
        ValueError: If code cannot be parsed or evaluated
    """
    # Build safe namespace with all SCATEX classes
    safe_namespace = {
        # Enums
        'Unit': Unit,
        'DayOfWeekType': DayOfWeekType,
        'MonthOfYearType': MonthOfYearType,
        'TimeOfDayType': TimeOfDayType,
        'Direction': Direction,
        # Core intervals
        'Year': Year,
        'Month': Month,
        'Day': Day,
        'Hour': Hour,
        'Minute': Minute,
        'Second': Second,
        'Instant': Instant,
        'Interval': Interval,
        # Repeating
        'Repeating': Repeating,
        'DayOfWeek': DayOfWeek,
        'MonthOfYear': MonthOfYear,
        'TimeOfDay': TimeOfDay,
        'RepeatingIntersection': RepeatingIntersection,
        # Relative operators
        'This': This,
        'Last': Last,
        'Next': Next,
        'Shift': Shift,
        'Period': Period,
        'Before': Before,
        'After': After,
        'Between': Between,
        # Composite patterns
        'ModifiedInterval': ModifiedInterval,
        'Approximate': Approximate,
        # Set operations
        'Union': Union,
        'Intersection': Intersection,
        # Convenience
        'Now': Now,
        'Today': Today,
        'Yesterday': Yesterday,
        'Tomorrow': Tomorrow,
        'Decade': Decade,
        'Century': Century,
        'Quarter': Quarter,
        'Unknown': Unknown,
        # Enum shortcuts for convenience
        'SECOND': Unit.SECOND,
        'MINUTE': Unit.MINUTE,
        'HOUR': Unit.HOUR,
        'DAY': Unit.DAY,
        'WEEK': Unit.WEEK,
        'MONTH': Unit.MONTH,
        'QUARTER': Unit.QUARTER,
        'YEAR': Unit.YEAR,
        'DECADE': Unit.DECADE,
        'CENTURY': Unit.CENTURY,
        'MONDAY': DayOfWeekType.MONDAY,
        'TUESDAY': DayOfWeekType.TUESDAY,
        'WEDNESDAY': DayOfWeekType.WEDNESDAY,
        'THURSDAY': DayOfWeekType.THURSDAY,
        'FRIDAY': DayOfWeekType.FRIDAY,
        'SATURDAY': DayOfWeekType.SATURDAY,
        'SUNDAY': DayOfWeekType.SUNDAY,
        'JANUARY': MonthOfYearType.JANUARY,
        'FEBRUARY': MonthOfYearType.FEBRUARY,
        'MARCH': MonthOfYearType.MARCH,
        'APRIL': MonthOfYearType.APRIL,
        'MAY': MonthOfYearType.MAY,
        'JUNE': MonthOfYearType.JUNE,
        'JULY': MonthOfYearType.JULY,
        'AUGUST': MonthOfYearType.AUGUST,
        'SEPTEMBER': MonthOfYearType.SEPTEMBER,
        'OCTOBER': MonthOfYearType.OCTOBER,
        'NOVEMBER': MonthOfYearType.NOVEMBER,
        'DECEMBER': MonthOfYearType.DECEMBER,
        'MORNING': TimeOfDayType.MORNING,
        'AFTERNOON': TimeOfDayType.AFTERNOON,
        'EVENING': TimeOfDayType.EVENING,
        'NIGHT': TimeOfDayType.NIGHT,
        'NOON': TimeOfDayType.NOON,
        'MIDNIGHT': TimeOfDayType.MIDNIGHT,
        'DAWN': TimeOfDayType.DAWN,
        'BEFORE': Direction.BEFORE,
        'AFTER': Direction.AFTER,
    }
    
    try:
        # Parse and evaluate the code
        expr = eval(code, {"__builtins__": {}}, safe_namespace)
        
        if isinstance(expr, TemporalExpression):
            return expr.evaluate(anchor)
        else:
            raise ValueError(f"Code did not produce a TemporalExpression: {type(expr)}")
            
    except SyntaxError as e:
        raise ValueError(f"Invalid SCATEX syntax: {e}")
    except NameError as e:
        raise ValueError(f"Unknown SCATEX operator: {e}")
    except Exception as e:
        raise ValueError(f"SCATEX evaluation error: {e}")


def format_interval(start: Optional[datetime], end: Optional[datetime]) -> str:
    """Format an interval as ISO 8601 strings."""
    start_str = start.isoformat() if start else "..."
    end_str = end.isoformat() if end else "..."
    return f"{start_str} / {end_str}"

