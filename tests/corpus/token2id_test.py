import os
from collections import Counter
from typing import Mapping

import pytest
from penelope.corpus import Token2Id
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN
from penelope.corpus.token2id import ClosedVocabularyError
from penelope.utility import path_add_suffix

TEST_TOKENS_STREAM1 = ['adam', 'anton', 'anton', 'beatrice', 'felicia', 'niklas', 'adam', 'adam']
TEST_TOKENS_STREAM2 = ['adam', 'anton', 'anton', 'beatrice', 'felicia', 'niklas', 'adam', 'adam', 'beata', 'beata']
EXPECTED_COUNTS2 = {'adam': 3, 'anton': 2, 'beatrice': 1, 'felicia': 1, 'niklas': 1, 'beata': 2, '__low-tf__': 1}


def tf_to_string(token2id: Token2Id) -> Mapping[str, int]:
    return {token2id.id2token[k]: n for k, n in dict(token2id.tf).items()}


def test_token2id_ingest():

    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM1)

    assert token2id.data == {'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4}
    assert token2id.tf is not None
    assert dict(token2id.tf) == {0: 3, 1: 2, 2: 1, 3: 1, 4: 1}
    assert tf_to_string(token2id) == {'adam': 3, 'anton': 2, 'beatrice': 1, 'felicia': 1, 'niklas': 1}


def test_dunders():
    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM1)
    assert 'adam' in token2id
    assert 'roger' not in token2id
    assert token2id['adam'] == 0
    assert len([x for x in token2id]) == len(set(TEST_TOKENS_STREAM1))


def test_token2id_insert_into_closed_vocabulary_raises_error():
    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM2).close()
    with pytest.raises(ClosedVocabularyError):
        token2id["roger"] = 99


def test_replace():

    tokens = ['a', 'a', 'b', 'c']

    ingested: Token2Id = Token2Id().ingest(tokens)

    token2id: Token2Id = Token2Id()
    token2id.replace(data={'a': 0, 'b': 1, 'c': 2}, tf=Counter({0: 2, 1: 1, 2: 1}))

    assert dict(token2id.data) == dict(ingested.data)
    assert dict(token2id.tf) == dict(ingested.tf)


def test_token2id_insert_into_closed_vocabulary_with_fallback_succeeds():
    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM2).close(fallback=GLOBAL_TF_THRESHOLD_MASK_TOKEN)
    assert token2id['word_that_dont_exists'] == token2id[GLOBAL_TF_THRESHOLD_MASK_TOKEN]


def test_token2id_find():

    token2id: Token2Id = Token2Id({'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4})

    assert set(token2id.find(what='adam')) == set([0])
    assert set(token2id.find(what='a*')) == set([0, 1])
    assert set(token2id.find(what=['a*', 'f*'])) == set([0, 1, 3])
    assert set(token2id.find(what=['a*', 'beatrice'])) == set([0, 1, 2])


def test_token2id_store_and_load():

    os.makedirs('./tests/output', exist_ok=True)

    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM1)

    filename = './tests/output/test_vocabulary.zip'
    tf_filename = path_add_suffix(filename, "_tf", new_extension=".pbz2")

    token2id.store(filename=filename)

    assert os.path.isfile(filename) and os.path.isfile(tf_filename)

    token2id_loaded: Token2Id = Token2Id.load(filename=filename)

    assert token2id_loaded is not None
    assert token2id_loaded.tf is not None

    assert token2id_loaded.data == {'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4}
    assert dict(token2id_loaded.tf) == {0: 3, 1: 2, 2: 1, 3: 1, 4: 1}


def test_token2id_compress_with_no_threshold_and_no_keeps_returns_self():

    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM2).close()

    assert token2id.data == {'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4, 'beata': 5}
    assert dict(token2id.tf) == {0: 3, 1: 2, 2: 1, 3: 1, 4: 1, 5: 2}

    token2id_compressed, translation = token2id.compress(tf_threshold=1, inplace=False, keeps=None)
    assert token2id_compressed is token2id
    assert translation is None


def test_token2id_inplace_compress_with_threshold_and_no_keeps():

    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM2).close()
    token2id_compressed, translation = token2id.compress(tf_threshold=2, inplace=False)
    assert dict(token2id_compressed.data) == {'adam': 0, 'anton': 1, 'beata': 2, GLOBAL_TF_THRESHOLD_MASK_TOKEN: 3}
    assert dict(token2id_compressed.tf) == {0: 3, 1: 2, 2: 2, 3: 3}
    assert translation == {0: 0, 1: 1, 5: 2}
    assert token2id.fallback_token is None
    assert token2id_compressed.fallback_token is not None
    assert token2id_compressed["roger"] == token2id_compressed[GLOBAL_TF_THRESHOLD_MASK_TOKEN]
    assert "roger" not in token2id_compressed


def test_token2id_compress_with_threshold_and_keeps_adds_masked_magic_token_with_correct_sum():

    mask_token: str = GLOBAL_TF_THRESHOLD_MASK_TOKEN
    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM2).close()
    token2id_compressed, _ = token2id.compress(tf_threshold=2, keeps={4}, inplace=False)

    assert mask_token not in token2id
    assert mask_token in token2id_compressed

    sum_of_masked_tokens = sum([v for k, v in EXPECTED_COUNTS2.items() if k not in token2id_compressed])

    assert token2id_compressed.tf[token2id_compressed[mask_token]] == sum_of_masked_tokens


def test_token2id_compress_with_ingested_mask_token_and_threshold_has_correct_magic_token_sum():
    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM2).ingest([GLOBAL_TF_THRESHOLD_MASK_TOKEN]).close()
    _, translation = token2id.compress(tf_threshold=2, inplace=True)
    assert dict(token2id.data) == {'adam': 0, 'anton': 1, 'beata': 2, GLOBAL_TF_THRESHOLD_MASK_TOKEN: 3}
    assert dict(token2id.tf) == {0: 3, 1: 2, 2: 2, 3: 4}
    assert translation == {0: 0, 1: 1, 5: 2, 6: 3}


def test_token2id_compress_with_threshold_and_keeps_scuccee3():
    token2id: Token2Id = Token2Id().ingest(TEST_TOKENS_STREAM2).close()
    _, translation = token2id.compress(tf_threshold=2, inplace=True, keeps={token2id["felicia"]})
    assert dict(token2id.data) == {'adam': 0, 'anton': 1, 'felicia': 2, 'beata': 3, GLOBAL_TF_THRESHOLD_MASK_TOKEN: 4}
    assert tf_to_string(token2id) == {'adam': 3, 'anton': 2, 'felicia': 1, 'beata': 2, '__low-tf__': 2}
    assert translation == {0: 0, 1: 1, 3: 2, 5: 3}

# def test_clip():
#     ...
#     data = {'*': 0, '__low-tf__': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}
#     tf = {0: 1, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 4}
#     token2id = Token2Id(data=data, tf=tf)
#     assert dict(token2id.data) == data
#     assert dict(token2id.tf) == tf

#     token2id.clip([0, 1, 2, 4, 6], inplace=True)
#     assert dict(token2id.data) == {'*': 0, '__low-tf__': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}
#     assert dict(token2id.tf) == {0: 1, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 4}


def test_translation():
    data = {'*': 0, '__low-tf__': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}
    tf = {0: 1, 1: 1, 2: 5, 3: 2, 4: 5, 5: 2, 6: 4}
    token2id = Token2Id(data=data, tf=tf)
    _, translation = token2id.compress(tf_threshold=4, inplace=True)
    tf = {0: 1, 1: 1, 2: 5, 3: 2, 4: 5, 5: 2, 6: 4}

    assert dict(token2id.data) == {'*': 0, '__low-tf__': 1, 'a': 2, 'c': 3, 'e': 4}
    assert dict(token2id.tf) ==  {0: 1, 1: 5, 2: 5, 3: 5, 4: 4}
    assert dict(translation) == {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}

def test_translate():

    data = {'*': 0, '__low-tf__': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}
    tf = {0: 1, 1: 1, 2: 5, 3: 2, 4: 5, 5: 2, 6: 4}
    ids_translation = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
    token2id = Token2Id(data=data, tf=tf)

    token2id.translate(ids_translation=ids_translation, inplace=True)

    assert dict(token2id.data) == {'*': 0, '__low-tf__': 1, 'a': 2, 'c': 3, 'e': 4}

    """Note that translate doesn't add LF-counts to LF-marker"""
    assert dict(token2id.tf) ==  {0: 1, 1: 1, 2: 5, 3: 5, 4: 4}

