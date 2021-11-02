import numpy as np
import pandas as pd
import pytest
from penelope.utility import PropertyValueMaskingOpts


def test_tagged_tokens_filter_opts_set_of_new_field_succeeds():
    filter_opts = PropertyValueMaskingOpts()
    filter_opts.is_stop = 1
    assert filter_opts.is_stop == 1


def test_tagged_tokens_filter_opts_get_of_unknown_field_succeeds():
    filter_opts = PropertyValueMaskingOpts()
    assert filter_opts.is_stop is None


def test_tagged_tokens_filter_props_is_as_expected():
    filter_opts = PropertyValueMaskingOpts()
    filter_opts.is_stop = 1
    filter_opts.pos_includes = ['NOUN', 'VERB']
    assert filter_opts.props == dict(is_stop=1, pos_includes=['NOUN', 'VERB'])


def test_tagged_tokens_filter_mask_when_boolean_attribute_succeeds():
    doc = pd.DataFrame(
        data=dict(
            text=['a', 'b', 'c', 'd'],
            is_stop=[True, False, True, np.nan],
            is_punct=[False, False, True, False],
        )
    )

    filter_opts = PropertyValueMaskingOpts(is_stop=True)
    mask = filter_opts.mask(doc)
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
