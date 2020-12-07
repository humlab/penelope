import pandas as pd
from penelope.corpus.document_index import (
    assert_is_monotonic_increasing_integer_series,
    document_index_upgrade,
    load_document_index,
)


def test_load_document_index():
    filename = './test_data/legal_instrument_index.csv'
    index = load_document_index(filename=filename, key_column=None, sep=';')
    assert isinstance(index, pd.DataFrame)

def test_assert_is_monotonic_increasing_integer_series():

    pass

def test_load_document_index_versions():

    filename = './tests/test_data/documents_index_doc_id.zip'

    document_index = pd.read_csv(filename, '\t', header=0, index_col=0, na_filter=False)

    document_index = document_index_upgrade(document_index)
    expected_columns = set(['filename', 'document_id', 'document_name', 'n_raw_tokens', 'n_tokens', 'n_terms'])
    assert set(document_index.columns.tolist()).intersection(expected_columns) == expected_columns

    # df, name = (document_index.rename_axis(''), 'document_index.csv')

    # file_utility.pandas_to_csv_zip(filename, (df, 'document_index'), extension="csv", sep='\t')
