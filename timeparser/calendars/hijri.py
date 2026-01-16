from timeparser.calendars import CalendarBase
from timeparser.calendars.hijri_parser import hijri_parser


class HijriCalendar(CalendarBase):
    parser = hijri_parser
