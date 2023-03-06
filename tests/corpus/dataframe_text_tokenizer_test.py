import unittest

import pandas as pd

import penelope.corpus.readers as readers

TEST_DATA_01 = [
    (2000, 'A B C', 'doc_01.txt'),
    (2000, 'B C D', 'doc_02.txt'),
    (2001, 'C B', 'doc_03.txt'),
    (2003, 'A B F', 'doc_04.txt'),
    (2003, 'E B', 'doc_05.txt'),
    (2003, 'F E E', 'doc_06.txt'),
]
TEST_DATA_02 = [
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


class Test_PandasCorpusReader(unittest.TestCase):
    def create_test_dataframe(self):
        df = pd.DataFrame(TEST_DATA_01, columns=['year', 'txt', 'filename'])
        return df

    def create_triple_meta_dataframe(self):
        df = pd.DataFrame(TEST_DATA_02, columns=['year', 'newspaper', 'txt'])
        return df

    def test_extract_metadata_when_sourcefile_has_year_and_newspaper(self):
        df = self.create_triple_meta_dataframe()
        df_m = df[[x for x in list(df.columns) if x != 'txt']]
        df_m['filename'] = df_m.index.astype(str)
        metadata = df_m.to_dict(orient='records')
        self.assertEqual(len(df), len(metadata))

    def test_reader_with_all_documents(self):
        df = self.create_test_dataframe()
        reader = readers.PandasCorpusReader(df)
        result = [x for x in reader]
        expected = [(name, doc.split()) for (_, doc, name) in TEST_DATA_01]

        self.assertEqual(expected, result)
        self.assertEqual([f'doc_0{i+1}.txt' for i in range(0, 6)], reader.filenames)
        self.assertEqual(
            [{'filename': name, 'year': year} for (year, _, name) in TEST_DATA_01],
            reader.metadata,
        )

    def test_reader_with_given_year(self):
        df = self.create_triple_meta_dataframe()

        expected_indices = [3, 4, 5, 9, 10, 11]

        reader = readers.PandasCorpusReader(df, year=2003)

        expected_filenames = [f'document_{i}.txt' for i in expected_indices]
        self.assertEqual(sorted(expected_filenames), sorted(reader.filenames))

        result = [x for x in reader]

        expected_name_docs = [(f'document_{i}.txt', TEST_DATA_02[i][2].split()) for i in expected_indices]
        self.assertEqual(sorted(expected_name_docs, key=lambda x: x[0]), result)

        self.assertEqual(
            sorted(
                [
                    dict(filename=f'document_{i}.txt', newspaper=TEST_DATA_02[i][1], year=TEST_DATA_02[i][0])
                    for i in expected_indices
                ],
                key=lambda x: x['filename'],
            ),
            reader.metadata,
        )
