"""Tests logger"""

from utils.logger import PloutosLogger

def test_logger_singleton():
    """Test pattern singleton"""
    logger1 = PloutosLogger()
    logger2 = PloutosLogger()
    
    assert logger1 is logger2

def test_logger_get():
    """Test récupération logger"""
    ploutos_logger = PloutosLogger()
    
    logger = ploutos_logger.get_logger('test')
    
    assert logger is not None
    assert 'ploutos.test' in logger.name

def test_logger_logging():
    """Test logging basique"""
    logger = PloutosLogger().get_logger('test')
    
    # Ne doit pas crash
    logger.info("Test info")
    logger.warning("Test warning")
    logger.debug("Test debug")
