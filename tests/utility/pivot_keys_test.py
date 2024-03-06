from io import StringIO

import pandas as pd
import pytest

from penelope.corpus import load_document_index
from penelope.utility import PivotKeys, PropertyValueMaskingOpts

# pylint: disable=redefined-outer-name

PIVOT_KEYS: dict = {
    'fågel': {
        'text_name': 'fågel',
        'id_name': 'fågel_id',
        'values': {
            'okänd': 0,
            'kråka': 1,
            'skata': 2,
        },
    },
    'husdjur': {
        'text_name': 'husdjur',
        'id_name': 'husdjur_id',
        'values': {
            'okänd': 0,
            'katt': 1,
            'hund': 2,
        },
    },
}


@pytest.fixture
def pivot_keys() -> PivotKeys:
    return PivotKeys(PIVOT_KEYS)


def test_pivot_keys_create(pivot_keys: PivotKeys):
    assert len(pivot_keys) == 2

    assert pivot_keys.key_name2key_id == {'fågel': 'fågel_id', 'husdjur': 'husdjur_id'}
    assert pivot_keys.key_id2key_name == {'fågel_id': 'fågel', 'husdjur_id': 'husdjur'}

    assert set(pivot_keys.text_names) == set({'fågel', 'husdjur'})
    assert set(pivot_keys.id_names) == set({'fågel_id', 'husdjur_id'})

    assert pivot_keys.is_satisfied()
    assert pivot_keys.get('fågel') == {
        'text_name': 'fågel',
        'id_name': 'fågel_id',
        'values': {
            'okänd': 0,
            'kråka': 1,
            'skata': 2,
        },
    }
    assert pivot_keys.key_value_name2id('fågel') == {
        'okänd': 0,
        'kråka': 1,
        'skata': 2,
    }


def test_pivot_keys_filter(pivot_keys: PivotKeys):
    opts: PropertyValueMaskingOpts = pivot_keys.create_filter_by_str_sequence(["fågel=kråka"], sep='=', decode=True)

    assert opts is not None
    assert opts.fågel_id == [1]

    opts: PropertyValueMaskingOpts = pivot_keys.create_filter_by_str_sequence(
        ["fågel=kråka", "husdjur=katt", "husdjur=hund"], sep='=', decode=False
    )

    assert opts is not None
    assert opts.fågel == ['kråka']
    assert set(opts.husdjur) == {'katt', 'hund'}


def test_pivot_keys_filter_with_sequence(pivot_keys: PivotKeys):
    opts: PropertyValueMaskingOpts = pivot_keys.create_filter_by_str_sequence(
        ["husdjur=katt,hund"], sep='=', decode=False, vsep=None
    )

    assert opts is not None
    assert set(opts.husdjur) == {'katt,hund'}

    opts: PropertyValueMaskingOpts = pivot_keys.create_filter_by_str_sequence(
        ["husdjur=katt,hund"], sep='=', decode=False, vsep=","
    )

    assert opts is not None
    assert set(opts.husdjur) == {"katt", "hund"}


def test_overload_document_index():

    document_index_str: str = """
filename;year;document_name;document_id;rating;n_raw_tokens
tran_2009_01_test.txt;2009;tran_2009_01_test;0;Good;44
tran_2009_02_test.txt;2009;tran_2009_02_test;1;Bad;59
tran_2019_01_test.txt;2019;tran_2019_01_test;2;Excellent;68
tran_2019_02_test.txt;2019;tran_2019_02_test;3;Good;59
tran_2019_03_test.txt;2019;tran_2019_03_test;4;Good;173
tran_2024_01_test.txt;2024;tran_2024_01_test;5;Bad;33
tran_2024_02_test.txt;2024;tran_2024_02_test;6;Excellent;44
tran_2029_01_test.txt;2029;tran_2029_01_test;7;Bad;24
tran_2029_02_test.txt;2029;tran_2029_02_test;8;Good;12
"""

    di: pd.DataFrame = load_document_index(filename=StringIO(document_index_str), sep=';')

    pivot_keys: PivotKeys = PivotKeys.create_by_index(di, 'rating')

    assert 'rating_id' in di.columns
    assert 'rating' in pivot_keys
    assert 'abc' not in pivot_keys
