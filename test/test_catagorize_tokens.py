import unittest
import re


class TestWordFeatures(unittest.TestCase):

    def test(self):
        comma_r = "\d{1,3}(,\d{3})*"
        careful_num_regex = re.compile("^" + "("+comma_r+"|\d+)" + "((\.\d+)|(s|st|th|nd|rd|km|m|bn|billion|k|million))?$")

        self.assertIsNotNone(re.fullmatch(comma_r, "90,000"))
        self.assertIsNotNone(re.fullmatch(comma_r, "90,000,112"))
        self.assertIsNone(re.fullmatch(comma_r, "90,000112"))
        self.assertIsNone(re.fullmatch(comma_r, "101,1"))
        self.assertIsNone(re.fullmatch(comma_r, "1,2,3"))
        self.assertIsNone(re.fullmatch(comma_r, "1,2,3"))

        self.assertIsNotNone(re.fullmatch(careful_num_regex, "90,000"))
        self.assertIsNotNone(re.fullmatch(careful_num_regex, "90,000.01"))
        self.assertIsNotNone(re.fullmatch(careful_num_regex, "91234.01"))
        self.assertIsNotNone(re.fullmatch(careful_num_regex, "91234st"))
        self.assertIsNotNone(re.fullmatch(careful_num_regex, "91,234th"))

