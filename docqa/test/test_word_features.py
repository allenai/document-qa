import unittest

from docqa.data_processing.text_features import extract_year, is_number, BasicWordFeatures


class TestWordFeatures(unittest.TestCase):
    def test_regex(self):
        fe = BasicWordFeatures()
        self.assertIsNone(fe.punc_regex.match("the"))
        self.assertIsNotNone(fe.punc_regex.match("!!!"))
        self.assertIsNotNone(fe.punc_regex.match("!.,()\"\'"))

    def test_non_english(self):
        tmp = BasicWordFeatures()
        self.assertIsNone(tmp.non_english.match("51,419,420"))
        self.assertIsNone(tmp.non_english.match("cat"))
        self.assertIsNone(tmp.non_english.match(","))
        self.assertIsNotNone(tmp.non_english.match("لجماهيري"))

    def test_date(self):
        self.assertEqual(extract_year("sdf"), None)
        self.assertEqual(extract_year("-1"), None)
        self.assertEqual(extract_year("1990"), 1990)
        self.assertEqual(extract_year("1990s"), 1990)
        self.assertEqual(extract_year("90s"), 1990)

    def test_any_num(self):
        tmp = BasicWordFeatures()
        self.assertIsNone(tmp.any_num_regex.match("cat"))
        self.assertIsNotNone(tmp.any_num_regex.match("c3at"))

    def test_numbers(self):
        self.assertIsNotNone(is_number("90,000"))
        self.assertIsNotNone(is_number("90,000,112"))
        self.assertIsNone(is_number("90,000112"))
        self.assertIsNone(is_number("101,1"))
        self.assertIsNone(is_number("1,2,3"))
        self.assertIsNone(is_number("1,2,3"))
        self.assertIsNone(is_number("0.1th"))

        self.assertIsNotNone(is_number("90,000"))
        self.assertIsNotNone(is_number("90,000.01"))
        self.assertIsNotNone(is_number("91234.01"))
        self.assertIsNotNone(is_number("91234st"))
        self.assertIsNotNone(is_number("91,234th"))
        self.assertIsNotNone(is_number(".034"))
        self.assertIsNotNone(is_number(".034km"))
