import operator
import os

import numpy as np
import pandas as pd
import pytest
from penelope.utility import (
    PropertyValueMaskingOpts,
    create_mask,
    list_filenames,
    pandas_read_csv_zip,
    pandas_to_csv_zip,
    try_split_column,
)
from tests.utils import OUTPUT_FOLDER


def test_try_split_column():

    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 1)
    assert df.columns.to_list() == ['A']

    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    df = try_split_column(df, 'B', '/', ['x', 'y'], False, 1)
    assert df.columns.to_list() == ['A']

    df = pd.DataFrame({'A': ['a/b', 'b/c', 'c']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 3)
    assert df.columns.to_list() == ['A']

    df = pd.DataFrame({'A': ['a/b', 'b/c', 'c']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 2)
    assert df.x.tolist() == ['a', 'b', 'c']
    assert df.y.tolist() == ['b', 'c', None]

    df = pd.DataFrame({'A': ['a/b', 'b/c', 'c/d']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 2)
    assert df.columns.to_list() == ['A', 'x', 'y']


def create_pandas_test_data():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}, index=[4, 5])
    df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'], index=[1, 2, 3])
    data = [(df1, 'df1.csv'), (df2, 'df2.csv')]
    return data


def test_pandas_to_csv_zip():

    filename = os.path.join(OUTPUT_FOLDER, "test_pandas_to_csv_zip.zip")
    data = create_pandas_test_data()

    pandas_to_csv_zip(filename, dfs=data, extension='csv', sep='\t')

    assert os.path.isfile(filename)
    assert set(list_filenames(zip_or_filename=filename, filename_pattern="*.csv")) == set({'df1.csv', 'df2.csv'})


def test_pandas_read_csv_zip():

    filename: str = os.path.join(OUTPUT_FOLDER, "test_pandas_to_csv_zip.zip")
    expected_data = create_pandas_test_data()
    pandas_to_csv_zip(filename, dfs=expected_data, extension='csv', sep='\t')

    data = pandas_read_csv_zip(filename, pattern='*.csv', sep='\t', index_col=0)

    assert 'df1.csv' in data and 'df2.csv' in data
    assert ((data['df1.csv'] == expected_data[0][0]).all()).all()
    assert ((data['df2.csv'] == expected_data[1][0]).all()).all()


def test_create_mask():

    df: pd.DataFrame = pd.DataFrame({'A': ['a', 'a', 'c', 'd'], 'B': [True, False, True, True], 'C': [1, 2, 3, 4]})

    mask: np.ndarray = create_mask(df, {})
    assert len(mask) == 4 and all(mask)

    empty: pd.DataFrame = df[df.A == 'X']
    mask: np.ndarray = create_mask(empty, {'A': 'a'})
    assert len(mask) == 0

    assert (create_mask(df, {'A': 'a'}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': (True, 'a')}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': (False, 'a')}) == [False, False, True, True]).all()
    assert (create_mask(df, {'A': (False, ['c', 'd'])}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': 'a', 'B': True}) == [True, False, False, False]).all()
    assert (create_mask(df, {'A': 'a', 'B': True, 'C': 2}) == [False, False, False, False]).all()

    assert (create_mask(df, {'A': (operator.eq, 'a')}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': (operator.gt, 'a')}) == [False, False, True, True]).all()
    assert (create_mask(df, {'A': ('gt', 'a')}) == [False, False, True, True]).all()
    assert (create_mask(df, {'A': (False, operator.gt, 'a')}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': (False, 'gt', 'a')}) == [True, True, False, False]).all()


def test_tagged_tokens_filter_opts_set_of_new_field_succeeds():
    masking_opts = PropertyValueMaskingOpts()
    masking_opts.is_stop = 1
    assert masking_opts.is_stop == 1


def test_tagged_tokens_filter_opts_get_of_unknown_field_succeeds():
    masking_opts = PropertyValueMaskingOpts()
    assert masking_opts.is_stop is None


def test_tagged_tokens_filter_props_is_as_expected():
    masking_opts = PropertyValueMaskingOpts()
    masking_opts.is_stop = 1
    masking_opts.pos_includes = ['NOUN', 'VERB']
    assert masking_opts.props == dict(is_stop=1, pos_includes=['NOUN', 'VERB'])


def test_tagged_tokens_filter_mask_when_boolean_attribute_succeeds():
    doc = pd.DataFrame(
        data=dict(
            text=['a', 'b', 'c', 'd'],
            is_stop=[True, False, True, np.nan],
            is_punct=[False, False, True, False],
        )
    )

    masking_opts = PropertyValueMaskingOpts(is_stop=True)
    mask = masking_opts.mask(doc)
    new_doc = doc[mask]
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'c']

    new_doc = doc[PropertyValueMaskingOpts(is_stop=None).mask(doc)]
    assert len(new_doc) == 4
    assert new_doc['text'].to_list() == ['a', 'b', 'c', 'd']

    new_doc = doc[PropertyValueMaskingOpts(is_stop=True, is_punct=True).mask(doc)]
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['c']

    new_doc = doc[PropertyValueMaskingOpts(is_stop=True, is_punct=False).mask(doc)]
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['a']

    new_doc = doc[PropertyValueMaskingOpts(is_stop=False).mask(doc)]
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['b']

    new_doc = doc[PropertyValueMaskingOpts(is_stop=[False]).mask(doc)]
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['b']


def test_tagged_tokens_filter_apply_when_boolean_attribute_succeeds():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], is_stop=[True, False, True]))

    new_doc = PropertyValueMaskingOpts(is_stop=True).apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'c']

    new_doc = PropertyValueMaskingOpts(is_stop=None).apply(doc)
    assert len(new_doc) == 3
    assert new_doc['text'].to_list() == ['a', 'b', 'c']

    new_doc = PropertyValueMaskingOpts(is_stop=False).apply(doc)
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['b']


def test_tagged_tokens_filter_apply_when_list_attribute_succeeds():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], pos=['X', 'X', 'Y']))

    new_doc = PropertyValueMaskingOpts(pos='X').apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'b']

    new_doc = PropertyValueMaskingOpts(pos=['X', 'Y']).apply(doc)
    assert len(new_doc) == 3
    assert new_doc['text'].to_list() == ['a', 'b', 'c']


def test_tagged_tokens_filter_apply_unknown_attribute_is_ignored():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], pos=['X', 'X', 'Y']))

    new_doc = PropertyValueMaskingOpts(kallekula='kurt').apply(doc)
    assert len(new_doc) == 3
    assert new_doc['text'].to_list() == ['a', 'b', 'c']


def test_tagged_tokens_filter_apply_when_unary_sign_operator_attribute_succeeds():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], pos=['X', 'X', 'Y']))

    new_doc = PropertyValueMaskingOpts(pos=(True, ['X'])).apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'b']

    new_doc = PropertyValueMaskingOpts(pos=(False, ['X'])).apply(doc)
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['c']

    new_doc = PropertyValueMaskingOpts(pos=(False, ['Y'])).apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'b']

    new_doc = PropertyValueMaskingOpts(pos=(False, ['X', 'Y'])).apply(doc)
    assert len(new_doc) == 0
    assert new_doc['text'].to_list() == []

    with pytest.raises(ValueError):
        new_doc = PropertyValueMaskingOpts(pos=(None, ['X', 'Y'])).apply(doc)

    assert len(PropertyValueMaskingOpts(pos=(True, 'X')).apply(doc)) == 2
    assert len(PropertyValueMaskingOpts(pos=(True, 0)).apply(doc)) == 0


def test_hot_attributes():

    doc = pd.DataFrame(
        data=dict(text=['a', 'b', 'c'], pos=['X', 'X', 'Y'], lemma=['a', 'b', 'c'], is_stop=[True, False, True])
    )

    assert len(PropertyValueMaskingOpts(pos=(True, 1)).hot_attributes(doc)) == 1
    assert len(PropertyValueMaskingOpts(pos='A', lemma='a').hot_attributes(doc)) == 2
    assert len(PropertyValueMaskingOpts(pos='A', lemma='a', _lemma='c').hot_attributes(doc)) == 2
    assert len(PropertyValueMaskingOpts().hot_attributes(doc)) == 0
    assert len(PropertyValueMaskingOpts(kalle=1, kula=2, kurt=2).hot_attributes(doc)) == 0
