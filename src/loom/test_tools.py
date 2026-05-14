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

    def test_ls_files(self):
        from loom.tools import _ls_files
        # Since we are in the root, it should list at least README.md
        output = _ls_files(".")
        self.assertIn("README.md", output)
