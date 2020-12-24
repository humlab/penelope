from io import StringIO

import numpy as np
import pandas as pd
import pytest
from penelope.corpus import (
    DocumentIndex,
    document_index_upgrade,
    load_document_index,
    load_document_index_from_str,
    update_document_index_properties,
)
from penelope.utility import assert_is_monotonic_increasing_integer_series, is_monotonic_increasing_integer_series

TEST_DOCUMENT_INDEX = """
;filename;year;year_id;document_name;document_id;title;n_raw_tokens
tran_2019_01_test;tran_2019_01_test.txt;2019;1;tran_2019_01_test;0;Even break;68
tran_2019_02_test;tran_2019_02_test.txt;2019;2;tran_2019_02_test;1;Night;59
tran_2019_03_test;tran_2019_03_test.txt;2019;3;tran_2019_03_test;2;Shining;173
tran_2020_01_test;tran_2020_01_test.txt;2020;1;tran_2020_01_test;3;Ostinato;33
tran_2020_02_test;tran_2020_02_test.txt;2020;2;tran_2020_02_test;4;Epilogue;44
"""

TEST_FAULTY_DOCUMENT_INDEX = """
filename;filename;year;year_id;document_name;document_id;title;n_raw_tokens
tran_2019_01_test.txt;tran_2019_01_test.txt;2019;1;tran_2019_01_test.txt;0;Even break;68
tran_2019_02_test.txt;tran_2019_02_test.txt;2019;2;tran_2019_02_test.txt;1;Night;59
tran_2019_03_test.txt;tran_2019_03_test.txt;2019;3;tran_2019_03_test.txt;2;Shining;173
tran_2020_01_test.txt;tran_2020_01_test.txt;2020;1;tran_2020_01_test.txt;3;Ostinato;33
tran_2020_02_test.txt;tran_2020_02_test.txt;2020;2;tran_2020_02_test.txt;4;Epilogue;44
"""
TEST_DOCUMENT_INDEX_WITH_GAPS_IN_YEAR = """
;filename;year;year_id;document_name;document_id;title;n_raw_tokens
tran_2019_01_test;tran_2019_01_test.txt;2019;1;tran_2019_01_test;0;Even break;68
tran_2019_02_test;tran_2019_02_test.txt;2019;2;tran_2019_02_test;1;Night;59
tran_2019_03_test;tran_2019_03_test.txt;2019;3;tran_2019_03_test;2;Shining;173
tran_2020_01_test;tran_2020_01_test.txt;2020;1;tran_2020_01_test;3;Ostinato;33
tran_2023_01_test;tran_2023_01_test.txt;2023;1;tran_2020_01_test;4;Epilogue;44
"""

TEST_DOCUMENT_INDEX2 = """
filename;year;year_id;document_name;document_id;title;n_raw_tokens
tran_2019_01_test.txt;2019;1;tran_2019_01_test;0;Even break;68
tran_2019_02_test.txt;2019;2;tran_2019_02_test;1;Night;59
tran_2019_03_test.txt;2019;3;tran_2019_03_test;2;Shining;173
tran_2020_01_test.txt;2020;1;tran_2020_01_test;3;Ostinato;33
tran_2020_02_test.txt;2020;2;tran_2020_02_test;4;Epilogue;44
"""


def load_test_index(data_str: str) -> DocumentIndex:
    index = DocumentIndex(load_document_index(filename=StringIO(data_str), key_column=None, sep=';'))
    return index


def test_store():
    index = load_test_index(TEST_DOCUMENT_INDEX)
    index.store('./tests/output/test_store_index.csv')
    with open('./tests/output/test_store_index.csv', 'r') as fp:
        data_str = fp.read()
    data_str.replace('\t', ';')
    assert TEST_DOCUMENT_INDEX[1:] == data_str


def test_load():

    index = load_document_index(filename=StringIO(TEST_FAULTY_DOCUMENT_INDEX), key_column=None, sep=';')
    assert isinstance(index, pd.DataFrame)
    assert len(index) == 5
    assert index.columns.tolist() == [
        'filename',
        'year',
        'year_id',
        'document_name',
        'document_id',
        'title',
        'n_raw_tokens',
    ]
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
        'n_raw_tokens',
    ]
    assert index2.document_id.tolist() == [0, 1, 2, 3, 4]
    assert index2.index.name == ''
    assert ((index == index2).all()).all()

    index3 = load_document_index_from_str(TEST_DOCUMENT_INDEX, key_column=None, sep=';')
    assert ((index == index3).all()).all()

    index4 = load_document_index_from_str(TEST_DOCUMENT_INDEX2, key_column=None, sep=';')
    assert ((index == index4).all()).all()


def test_load_versions():
    filename = './tests/test_data/documents_index_doc_id.zip'
    document_index = pd.read_csv(filename, '\t', header=0, index_col=0, na_filter=False)
    document_index = document_index_upgrade(document_index)
    expected_columns = set(['filename', 'document_id', 'document_name', 'n_raw_tokens', 'n_tokens', 'n_raw_tokens'])
    assert set(document_index.columns.tolist()).intersection(expected_columns) == expected_columns


def test_from_metadata():
    pass


def test_from_str():
    pass


def test_consolidate():
    pass


def test_upgrade():
    filename = './tests/test_data/documents_index_doc_id.zip'
    document_index = pd.read_csv(filename, '\t', header=0, index_col=0, na_filter=False)
    document_index = DocumentIndex(document_index).upgrade()
    expected_columns = set(['filename', 'document_id', 'document_name', 'n_raw_tokens', 'n_tokens', 'n_raw_tokens'])
    assert set(document_index.columns.tolist()).intersection(expected_columns) == expected_columns


def test_update_counts():
    index = load_document_index(filename=StringIO(TEST_FAULTY_DOCUMENT_INDEX), key_column=None, sep=';')

    statistics = {'extra_1': 1, 'extra_2': 2}

    assert 'extra_1' not in index.columns
    update_document_index_properties(index, document_name='tran_2020_01_test', property_bag=statistics)

    assert 'extra_1' in index.columns
    assert 'extra_2' in index.columns

    assert int(index.loc['tran_2020_01_test'].extra_1) == 1
    assert int(index.loc['tran_2020_01_test'].extra_2) == 2

    assert index.extra_1.sum() == 1
    assert index.extra_2.sum() == 2

    statistics = {'extra_1': 10, 'extra_2': 22}
    update_document_index_properties(index, document_name='tran_2020_01_test', property_bag=statistics)

    assert int(index.loc['tran_2020_01_test'].extra_1) == 10
    assert int(index.loc['tran_2020_01_test'].extra_2) == 22

    assert index.extra_1.sum() == 10
    assert index.extra_2.sum() == 22

    statistics = {'extra_1': 10, 'extra_2': 22}
    update_document_index_properties(index, document_name='tran_2020_01_test', property_bag=statistics)


def test_add_attributes():
    pass


def test_update_properties():
    pass


def test_collapse_by_integer_column():
    index: pd.DataFrame = load_document_index(filename=StringIO(TEST_DOCUMENT_INDEX), key_column=None, sep=';')
    result: pd.DataFrame = DocumentIndex(index).group_by_column(column_name='year', transformer=None, index_values=None)

    assert result.category.tolist() == [2019, 2022]
    assert result.year.tolist() == [2019, 2022]


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
