import unittest

from data_processing.text_features import extract_year, is_number


class TestWordFeatures(unittest.TestCase):
    def test_regex(self):
        fe = SurfaceWordFeatures()
        self.assertIsNone(fe.punc_regex.match("the"))
        self.assertIsNotNone(fe.punc_regex.match("!!!"))
        self.assertIsNotNone(fe.punc_regex.match("!.,()\"\'"))

        self.assertIsNone(fe.num_regex.match("tge"))
        self.assertIsNone(fe.num_regex.match("!!,!"))
        self.assertIsNotNone(fe.num_regex.match("19"))
        self.assertIsNotNone(fe.num_regex.match("1,9"))
        self.assertIsNotNone(fe.num_regex.match("1,9.001"))

    def test_parse_num(self):
        tmp = DeepSurfaceWordFeatures()
        self.assertEqual(tmp.robust_parse_num("51,419,420"), 51419420)
        self.assertEqual(tmp.robust_parse_num("1,051,419,420"), 1051419420)
        self.assertEqual(tmp.robust_parse_num("419,420"), 419420)
        self.assertIsNone(tmp.robust_parse_num("51,419,42"))
        self.assertIsNone(tmp.robust_parse_num("51,419,42a"))

    def test_non_english(self):
        tmp = DeepSurfaceWordFeatures()
        self.assertIsNotNone(tmp.english.match("51,419,420"))
        self.assertIsNotNone(tmp.english.match("cat"))
        self.assertIsNotNone(tmp.english.match(","))
        self.assertIsNone(tmp.english.match("لجماهيري"))

    def test_date(self):
        self.assertEqual(extract_year("sdf"), None)
        self.assertEqual(extract_year("-1"), None)
        self.assertEqual(extract_year("1990"), 1990)
        self.assertEqual(extract_year("1990s"), 1990)
        self.assertEqual(extract_year("90s"), 1990)

    def test_any_num(self):
        tmp = DeepSurfaceWordFeatures()
        self.assertIsNone(tmp.any_num_regex.match("cat"))
        self.assertIsNotNone(tmp.any_num_regex.match("c3at"))

        self.assertIsNotNone(tmp.any_non_english.match("cat"))
        self.assertIsNotNone(tmp.any_non_english.match("134.3"))
        self.assertIsNotNone(tmp.any_non_english.match(":"))
        self.assertIsNone(tmp.any_non_english.match("لجماهيري"))
        self.assertIsNotNone(tmp.any_non_english.match("لجماهيريadsf"))

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

    def test_word_features(self):
        fe = SurfaceWordFeatures()
        arr = fe.get_word_features("The")
        self.assertEqual(
            list(arr),
            [1, 3, 0, 0, 0])

        arr = fe.get_word_features("\"")
        self.assertEqual(
            list(arr),
            [0, 1, 0, 1, 1])

    def test_prev_features(self):
        fe = SurfaceWordFeatures()
        arr = fe.get_sentence_features(["The", "Cat", ".", "Then", "a"])
        self.assertEqual(arr[0, -2], 0)
        self.assertEqual(arr[1, -2], 1)
        self.assertEqual(arr[3, -2], 0)

        self.assertEqual(arr[0, -1], 0)
        self.assertEqual(arr[1, -1], 1)
        self.assertEqual(arr[3, -1], 0)
        self.assertEqual(arr[4, -1], 0)

