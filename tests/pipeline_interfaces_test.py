import pandas as pd
import pytest
from penelope.corpus.readers import TaggedTokensFilterOpts


def test_tagged_tokens_filter_opts_set_of_new_field_succeeds():
    filter_opts = TaggedTokensFilterOpts()
    filter_opts.is_stop = 1
    assert filter_opts.is_stop == 1


def test_tagged_tokens_filter_opts_get_of_unknown_field_succeeds():
    filter_opts = TaggedTokensFilterOpts()
    assert filter_opts.is_stop is None


def test_tagged_tokens_filter_props_is_as_expected():
    filter_opts = TaggedTokensFilterOpts()
    filter_opts.is_stop = 1
    filter_opts.pos_includes = ['NOUN', 'VERB']
    assert filter_opts.props == dict(is_stop=1, pos_includes=['NOUN', 'VERB'])


def test_tagged_tokens_filter_mask_when_boolean_attribute_succeeds():
    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], is_stop=[True,False,True]))

    filter_opts = TaggedTokensFilterOpts(is_stop=True)
    mask = filter_opts.mask(doc)
    new_doc = doc[mask]
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'c']

    new_doc = doc[TaggedTokensFilterOpts(is_stop=None).mask(doc)]
    assert len(new_doc) == 3
    assert new_doc['text'].to_list() == ['a', 'b', 'c']


    new_doc = doc[TaggedTokensFilterOpts(is_stop=False).mask(doc)]
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['b']

def test_tagged_tokens_filter_apply_when_boolean_attribute_succeeds():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], is_stop=[True,False,True]))

    new_doc = TaggedTokensFilterOpts(is_stop=True).apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'c']

    new_doc = TaggedTokensFilterOpts(is_stop=None).apply(doc)
    assert len(new_doc) == 3
    assert new_doc['text'].to_list() == ['a', 'b', 'c']

    new_doc = TaggedTokensFilterOpts(is_stop=False).apply(doc)
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['b']

def test_tagged_tokens_filter_apply_when_list_attribute_succeeds():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], pos=['X','X','Y']))

    new_doc = TaggedTokensFilterOpts(pos='X').apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'b']

    new_doc = TaggedTokensFilterOpts(pos=['X', 'Y']).apply(doc)
    assert len(new_doc) == 3
    assert new_doc['text'].to_list() == ['a', 'b', 'c']

def test_tagged_tokens_filter_apply_unknown_attribute_is_ignored():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], pos=['X','X','Y']))

    new_doc = TaggedTokensFilterOpts(kallekula='kurt').apply(doc)
    assert len(new_doc) == 3
    assert new_doc['text'].to_list() == ['a', 'b', 'c']

def test_tagged_tokens_filter_apply_when_unary_sign_operator_attribute_succeeds():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], pos=['X','X','Y']))

    new_doc = TaggedTokensFilterOpts(pos=(True,['X'])).apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a', 'b']

    new_doc = TaggedTokensFilterOpts(pos=(False,['X'])).apply(doc)
    assert len(new_doc) == 1
    assert new_doc['text'].to_list() == ['c']

    new_doc = TaggedTokensFilterOpts(pos=(False,['Y'])).apply(doc)
    assert len(new_doc) == 2
    assert new_doc['text'].to_list() == ['a','b']

    new_doc = TaggedTokensFilterOpts(pos=(False, ['X', 'Y'])).apply(doc)
    assert len(new_doc) == 0
    assert new_doc['text'].to_list() == []

    with pytest.raises(ValueError):
        new_doc = TaggedTokensFilterOpts(pos=(None, ['X', 'Y'])).apply(doc)

    with pytest.raises(ValueError):
        new_doc = TaggedTokensFilterOpts(pos=(True, 'X')).apply(doc)

    with pytest.raises(ValueError):
        new_doc = TaggedTokensFilterOpts(pos=(True, 1)).apply(doc)

def test_hot_attributes():

    doc = pd.DataFrame(data=dict(text=['a', 'b', 'c'], pos=['X','X','Y'], lemma=['a', 'b', 'c'], is_stop=[True, False, True]))

    assert len(TaggedTokensFilterOpts(pos=(True, 1)).hot_attributes(doc)) == 1
    assert len(TaggedTokensFilterOpts(pos='A', lemma='a').hot_attributes(doc)) == 2
    assert len(TaggedTokensFilterOpts(pos='A', lemma='a', _lemma='c').hot_attributes(doc)) == 2
    assert len(TaggedTokensFilterOpts().hot_attributes(doc)) == 0
    assert len(TaggedTokensFilterOpts(kalle=1, kula=2, kurt=2).hot_attributes(doc)) == 0

