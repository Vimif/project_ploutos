import logging

from core.utils import format_duration, setup_logging, timestamp


def test_setup_logging():
    logger = setup_logging("test_logger", "test.log")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_format_duration():
    assert format_duration(3665) == "01:01:05"
    assert format_duration(60) == "00:01:00"


def test_timestamp():
    ts = timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 15  # YYYYMMDD_HHMMSS
