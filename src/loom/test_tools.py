import unittest
from loom.tools import _truncate

class TestTools(unittest.TestCase):
    def test_truncate(self):
        self.assertEqual(_truncate("short"), "short")
        
        long_str = "a" * 20000
        result = _truncate(long_str)
        self.assertTrue(len(result) < 20000)
        self.assertIn("truncated", result)

if __name__ == "__main__":
    unittest.main()
