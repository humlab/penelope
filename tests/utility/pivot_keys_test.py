import pytest

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
