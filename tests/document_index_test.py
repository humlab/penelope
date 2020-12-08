from io import StringIO

import numpy as np
import pandas as pd
import pytest
from penelope.corpus.document_index import (
    assert_is_monotonic_increasing_integer_series,
    document_index_upgrade,
    is_monotonic_increasing_integer_series,
    load_document_index,
    load_document_index_from_str,
    update_document_index_statistics,
)

TEST_FAULTY_DOCUMENT_INDEX = """
filename;filename;year;year_id;document_name;document_id;title;n_terms
tran_2019_01_test.txt;tran_2019_01_test.txt;2019;1;tran_2019_01_test.txt;0;Even break;68
tran_2019_02_test.txt;tran_2019_02_test.txt;2019;2;tran_2019_02_test.txt;1;Night;59
tran_2019_03_test.txt;tran_2019_03_test.txt;2019;3;tran_2019_03_test.txt;2;Shining;173
tran_2020_01_test.txt;tran_2020_01_test.txt;2020;1;tran_2020_01_test.txt;3;Ostinato;33
tran_2020_02_test.txt;tran_2020_02_test.txt;2020;2;tran_2020_02_test.txt;4;Epilogue;44
"""

TEST_DOCUMENT_INDEX = """
;filename;year;year_id;document_name;document_id;title;n_terms
tran_2019_01_test;tran_2019_01_test.txt;2019;1;tran_2019_01_test;0;Even break;68
tran_2019_02_test;tran_2019_02_test.txt;2019;2;tran_2019_02_test;1;Night;59
tran_2019_03_test;tran_2019_03_test.txt;2019;3;tran_2019_03_test;2;Shining;173
tran_2020_01_test;tran_2020_01_test.txt;2020;1;tran_2020_01_test;3;Ostinato;33
tran_2020_02_test;tran_2020_02_test.txt;2020;2;tran_2020_02_test;4;Epilogue;44
"""

TEST_DOCUMENT_INDEX2 = """
filename;year;year_id;document_name;document_id;title;n_terms
tran_2019_01_test.txt;2019;1;tran_2019_01_test;0;Even break;68
tran_2019_02_test.txt;2019;2;tran_2019_02_test;1;Night;59
tran_2019_03_test.txt;2019;3;tran_2019_03_test;2;Shining;173
tran_2020_01_test.txt;2020;1;tran_2020_01_test;3;Ostinato;33
tran_2020_02_test.txt;2020;2;tran_2020_02_test;4;Epilogue;44
"""


def test_load_document_index():

    index = load_document_index(filename=StringIO(TEST_FAULTY_DOCUMENT_INDEX), key_column=None, sep=';')
    assert isinstance(index, pd.DataFrame)
    assert len(index) == 5
    assert index.columns.tolist() == ['filename', 'year', 'year_id', 'document_name', 'document_id', 'title', 'n_terms']
    assert index.document_id.tolist() == [0, 1, 2, 3, 4]
    assert index.index.name == ''

    index2 = load_document_index(filename=StringIO(TEST_DOCUMENT_INDEX), key_column=None, sep=';')
    assert isinstance(index2, pd.DataFrame)
    assert len(index2) == 5
    assert index2.columns.tolist() == [
        'filename',
        'year',
        'year_id',
        'document_name',
        'document_id',
        'title',
        'n_terms',
    ]
    assert index2.document_id.tolist() == [0, 1, 2, 3, 4]
    assert index2.index.name == ''
    assert ((index == index2).all()).all()

    index3 = load_document_index_from_str(TEST_DOCUMENT_INDEX, key_column=None, sep=';')
    assert ((index == index3).all()).all()

    index4 = load_document_index_from_str(TEST_DOCUMENT_INDEX2, key_column=None, sep=';')
    assert ((index == index4).all()).all()


def test_assert_is_monotonic_increasing_integer_series():
    assert_is_monotonic_increasing_integer_series(pd.Series([0, 1, 2], dtype=np.int))
    with pytest.raises(ValueError):
        assert_is_monotonic_increasing_integer_series(pd.Series([0, -1, 2], dtype=np.int))
    with pytest.raises(ValueError):
        assert_is_monotonic_increasing_integer_series(pd.Series(['a', 'b', 'c']))


def test_is_monotonic_increasing_integer_series():
    assert is_monotonic_increasing_integer_series(pd.Series([0, 1, 2], dtype=np.int))
    assert not is_monotonic_increasing_integer_series(pd.Series([0, -1, 2], dtype=np.int))
    assert not is_monotonic_increasing_integer_series(pd.Series(['a', 'b', 'c']))


def test_load_document_index_versions():
    filename = './tests/test_data/documents_index_doc_id.zip'
    document_index = pd.read_csv(filename, '\t', header=0, index_col=0, na_filter=False)
    document_index = document_index_upgrade(document_index)
    expected_columns = set(['filename', 'document_id', 'document_name', 'n_raw_tokens', 'n_tokens', 'n_terms'])
    assert set(document_index.columns.tolist()).intersection(expected_columns) == expected_columns


# def metadata_to_document_index(metadata: List[Dict], *, document_id_field: str = None) -> pd.DataFrame:
#     """Creates a document index with all collected metadata key as columns.
#     The data frame's index is set to `filename` by (default) or `index_field` if it is supplied.
#     The index is also added as a `document_id` column."""

#     if metadata is None or len(metadata) == 0:
#         metadata = {'filename': [], 'document_id': []}

#     catalogue: pd.DataFrame = pd.DataFrame(metadata)

#     if 'filename' not in catalogue.columns:
#         raise ValueError("metadata is missing mandatory field `filename`")

#     if 'document_id' in catalogue.columns:
#         logging.warning("filename metadata already has a column named `document_id` (will be overwritten)")

#     if 'document_name' not in catalogue.columns:
#         catalogue['document_name'] = catalogue.filename.apply(strip_path_and_extension)

#     catalogue['document_id'] = catalogue[document_id_field] if document_id_field is not None else catalogue.index

#     assert_is_monotonic_increasing_integer_series(catalogue['document_id'])

#     catalogue = catalogue.set_index('document_name', drop=False)

#     return catalogue


# def store_document_index(document_index: pd.DataFrame, filename: str):
#     document_index.to_csv(filename, sep='\t', header=True)


# def consolidate_document_index(document_index: pd.DataFrame, reader_index: pd.DataFrame):
#     """Returns a consolidated document index from an existing index, if exists,
#     and the reader index."""

#     if document_index is not None:
#         columns = [x for x in reader_index.columns if x not in document_index.columns]
#         if len(columns) > 0:
#             document_index = document_index.merge(reader_index[columns], left_index=True, right_index=True, how='left')
#         return document_index

#     return reader_index


# def document_index_upgrade(document_index: pd.DataFrame) -> pd.DataFrame:
#     """Fixes older versions of document indexes"""

#     if 'document_name' not in document_index.columns:
#         document_index['document_name'] = document_index.filename.apply(strip_path_and_extension)

#     if document_index.index.dtype == np.dtype('int64'):

#         if 'document_id' not in document_index.columns:
#             document_index['document_id'] = document_index.index

#     document_index = document_index.set_index('document_name', drop=False).rename_axis('')

#     return document_index


# def add_document_index_attributes(*, catalogue: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
#     """ Adds document meta data to given data frame (must have a document_id) """
#     df = target.merge(catalogue, how='inner', left_on='document_id', right_on='document_id')
#     return df


# def update_document_index_token_counts(
#     document_index: pd.DataFrame, doc_token_counts: List[Tuple[str, int, int]]
# ) -> pd.DataFrame:
#     """Updates or adds fields `n_raw_tokens` and `n_tokens` to document index from collected during a corpus read pass
#     Only updates values that don't already exist in the document index"""
#     try:

#         strip_ext = lambda filename: os.path.splitext(filename)[0]

#         df_counts: pd.DataFrame = pd.DataFrame(data=doc_token_counts, columns=['filename', 'x_raw_tokens', 'x_tokens'])
#         df_counts['document_name'] = df_counts.filename.apply(strip_ext)
#         df_counts = df_counts.set_index('document_name').rename_axis('').drop('filename', axis=1)

#         if 'document_name' not in document_index.columns:
#             document_index['document_name'] = document_index.filename.apply(strip_ext)

#         if 'n_raw_tokens' not in document_index.columns:
#             document_index['n_raw_tokens'] = np.nan

#         if 'n_tokens' not in document_index.columns:
#             document_index['n_tokens'] = np.nan

#         document_index = document_index.merge(df_counts, how='left', left_index=True, right_index=True)

#         document_index['n_raw_tokens'] = document_index['x_raw_tokens'].fillna(document_index['n_raw_tokens'])
#         document_index['n_tokens'] = document_index['x_tokens'].fillna(document_index['n_tokens'])

#         document_index = document_index.drop(['x_raw_tokens', 'x_tokens'], axis=1)

#     except Exception as ex:
#         logging.error(ex)

#     return document_index


# def test_update_document_index_statistics(document_index, document_name: str, statistics: Mapping[str, int]):
#     new_keys = [ key for key in statistics if key not in document_index.columns ]
#     for key in new_keys:
#         document_index.insert(len(document_index.columns), key, np.nan)
#     document_index.update(pd.DataFrame(data=statistics, index=[document_name]))


def test_update_document_index_statistics():
    index = load_document_index(filename=StringIO(TEST_FAULTY_DOCUMENT_INDEX), key_column=None, sep=';')

    statistics = {'extra_1': 1, 'extra_2': 2}

    assert 'extra_1' not in index.columns
    update_document_index_statistics(index, document_name='tran_2020_01_test', statistics=statistics)

    assert 'extra_1' in index.columns
    assert 'extra_2' in index.columns

    assert int(index.loc['tran_2020_01_test'].extra_1) == 1
    assert int(index.loc['tran_2020_01_test'].extra_2) == 2

    assert index.extra_1.sum() == 1
    assert index.extra_2.sum() == 2

    statistics = {'extra_1': 10, 'extra_2': 22}
    update_document_index_statistics(index, document_name='tran_2020_01_test', statistics=statistics)

    assert int(index.loc['tran_2020_01_test'].extra_1) == 10
    assert int(index.loc['tran_2020_01_test'].extra_2) == 22

    assert index.extra_1.sum() == 10
    assert index.extra_2.sum() == 22

    statistics = {'extra_1': 10, 'extra_2': 22}
    update_document_index_statistics(index, document_name='tran_2020_01_test', statistics=statistics)
