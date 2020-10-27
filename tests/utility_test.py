import pytest

import penelope.utility as utility


@pytest.mark.parametrize(
    'lst1, lst2, key, expected',
    [
        ([{'b': 2}], [{'c': 3}], 'a', [{'b': 2}]),
        ([{'a': 1, 'b': 2}], [{'a': 1, 'c': 3}], 'a', [{'a': 1, 'b': 2, 'c': 3}]),
        ([{'a': 1, 'b': 2}], [{'c': 3}], 'a', [{'a': 1, 'b': 2}]),
    ],
)
def test_lists_of_dicts_merged_by_key_when_results_are_expected(lst1, lst2, key, expected):

    result = utility.lists_of_dicts_merged_by_key(lst1, lst2, key=key)
    assert expected == result


@pytest.mark.parametrize(
    'lst1, lst2, key',
    [
        ([{'b': 2}], [{'a': 1, 'c': 3}], 'a'),
    ],
)
def test_lists_of_dicts_merged_by_key_when_exception_is_expected(lst1, lst2, key):

    with pytest.raises(ValueError):
        _ = utility.lists_of_dicts_merged_by_key(lst1, lst2, key=key)


#    metadata_lookup =  {x['filename']: x for x in utility.lists_of_dicts_merged_by_key(reader.metadata, extra_metadata, key='filename')}
