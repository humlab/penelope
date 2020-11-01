import unittest

import pandas as pd
import penelope.corpus.readers as readers


class Test_DataFrameTextTokenizer(unittest.TestCase):
    def create_test_dataframe(self):
        data = [(2000, 'A B C'), (2000, 'B C D'), (2001, 'C B'), (2003, 'A B F'), (2003, 'E B'), (2003, 'F E E')]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        return df

    def create_triple_meta_dataframe(self):
        data = [
            (2000, 'AB', 'A B C'),
            (2000, 'AB', 'B C D'),
            (2001, 'AB', 'C B'),
            (2003, 'AB', 'A B F'),
            (2003, 'AB', 'E B'),
            (2003, 'AB', 'F E E'),
            (2000, 'EX', 'A B C'),
            (2000, 'EX', 'B C D'),
            (2001, 'EX', 'C B'),
            (2003, 'EX', 'A A B'),
            (2003, 'EX', 'B B'),
            (2003, 'EX', 'A E'),
        ]
        df = pd.DataFrame(data, columns=['year', 'newspaper', 'txt'])
        return df

    def test_extract_metadata_when_sourcefile_has_year_and_newspaper(self):
        df = self.create_triple_meta_dataframe()
        df_m = df[[x for x in list(df.columns) if x != 'txt']]
        df_m['filename'] = df_m.index.astype(str)
        metadata = df_m.to_dict(orient='records')
        self.assertEqual(len(df), len(metadata))

    def test_reader_with_all_documents(self):
        df = self.create_test_dataframe()
        reader = readers.DataFrameTextTokenizer(df)
        result = [x for x in reader]
        expected = [
            ('0', 'A B C'.split()),
            ('1', 'B C D'.split()),
            ('2', 'C B'.split()),
            ('3', 'A B F'.split()),
            ('4', 'E B'.split()),
            ('5', 'F E E'.split()),
        ]
        self.assertEqual(expected, result)
        self.assertEqual(['0', '1', '2', '3', '4', '5'], reader.filenames)
        self.assertEqual(
            [
                dict(filename='0', year=2000),
                dict(filename='1', year=2000),
                dict(filename='2', year=2001),
                dict(filename='3', year=2003),
                dict(filename='4', year=2003),
                dict(filename='5', year=2003),
            ],
            reader.metadata,
        )

    def test_reader_with_given_year(self):
        df = self.create_triple_meta_dataframe()
        reader = readers.DataFrameTextTokenizer(df, year=2003)
        result = [x for x in reader]
        expected = [
            ('3', ['A', 'B', 'F']),
            ('4', ['E', 'B']),
            ('5', ['F', 'E', 'E']),
            ('9', ['A', 'A', 'B']),
            ('10', ['B', 'B']),
            ('11', ['A', 'E']),
        ]
        self.assertEqual(expected, result)
        self.assertEqual(['3', '4', '5', '9', '10', '11'], reader.filenames)
        self.assertEqual(
            [
                dict(filename='3', newspaper='AB', year=2003),
                dict(filename='4', newspaper='AB', year=2003),
                dict(filename='5', newspaper='AB', year=2003),
                dict(filename='9', newspaper='EX', year=2003),
                dict(filename='10', newspaper='EX', year=2003),
                dict(filename='11', newspaper='EX', year=2003),
            ],
            reader.metadata,
        )
