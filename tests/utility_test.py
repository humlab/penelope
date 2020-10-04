import unittest

import penelope.utility as utility


class test_Utility(unittest.TestCase):
    def setUp(self):
        pass

    def test_dehypen(self):

        text = 'absdef\n'
        result = utility.fix_hyphenation(text)
        self.assertEqual(text, result)

        text = 'abs-def\n'
        result = utility.fix_hyphenation(text)
        self.assertEqual(text, result)

        text = 'abs - def\n'
        result = utility.fix_hyphenation(text)
        self.assertEqual(text, result)

        text = 'abs-\ndef'
        result = utility.fix_hyphenation(text)
        self.assertEqual('absdef\n', result)

        text = 'abs- \r\n def'
        result = utility.fix_hyphenation(text)
        self.assertEqual('absdef\n', result)

    def test_fix_whitespaces(self):

        text = 'absdef\n'
        result = utility.fix_whitespaces(text)
        self.assertEqual('absdef', result)

        text = ' absdef \n'
        result = utility.fix_whitespaces(text)
        self.assertEqual('absdef', result)

        text = 'abs  def'
        result = utility.fix_whitespaces(text)
        self.assertEqual('abs def', result)

        text = 'abs\n def'
        result = utility.fix_whitespaces(text)
        self.assertEqual('abs def', result)

        text = 'abs- \r\n def'
        result = utility.fix_whitespaces(text)
        self.assertEqual('abs- def', result)
