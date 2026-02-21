import unittest
from core.utils import setup_logging, format_duration, timestamp

class TestUtils(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(format_duration(3661), "01:01:01")
        self.assertEqual(format_duration(60), "00:01:00")

    def test_timestamp(self):
        ts = timestamp()
        self.assertRegex(ts, r"\d{8}_\d{6}")

    def test_setup_logging(self):
        logger = setup_logging("test_logger")
        self.assertTrue(logger.hasHandlers())

if __name__ == '__main__':
    unittest.main()
