import os

import pytest
from penelope.corpus import Token2Id
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN
from penelope.corpus.token2id import ClosedVocabularyError
from penelope.utility import path_add_suffix


def test_token2id_find():

    token2id: Token2Id = Token2Id({'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4})

    assert set(token2id.find(what='adam')) == set([0])
    assert set(token2id.find(what='a*')) == set([0, 1])
    assert set(token2id.find(what=['a*', 'f*'])) == set([0, 1, 3])
    assert set(token2id.find(what=['a*', 'beatrice'])) == set([0, 1, 2])
    assert token2id.tf is None


def test_token2id_ingest():
    os.makedirs('./tests/output', exist_ok=True)

    tokens = ['adam', 'anton', 'anton', 'beatrice', 'felicia', 'niklas', 'adam', 'adam']
    token2id: Token2Id = Token2Id().ingest(tokens)

    assert token2id.data == {'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4}

    assert token2id.tf is not None

    assert token2id.tf[0] == 3
    assert token2id.tf[1] == 2
    assert token2id.tf[2] == 1
    assert token2id.tf[3] == 1
    assert token2id.tf[4] == 1

    assert token2id.tf[token2id['adam']] == 3
    assert token2id.tf[token2id['anton']] == 2
    assert token2id.tf[token2id['beatrice']] == 1
    assert token2id.tf[token2id['felicia']] == 1
    assert token2id.tf[token2id['niklas']] == 1

    filename = './tests/output/test_vocabulary.zip'
    tf_filename = path_add_suffix(filename, "_tf", new_extension=".pbz2")

    token2id.store(filename=filename)

    assert os.path.isfile(filename)
    assert os.path.isfile(tf_filename)

    token2id_loaded: Token2Id = Token2Id.load(filename=filename)

    assert token2id_loaded is not None
    assert token2id_loaded.tf is not None

    assert token2id_loaded.data == {'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4}
    assert token2id_loaded.tf[0] == 3
    assert token2id_loaded.tf[1] == 2
    assert token2id_loaded.tf[2] == 1
    assert token2id_loaded.tf[3] == 1
    assert token2id_loaded.tf[4] == 1


def test_token2id_compress():

    tokens = [
        'adam',
        'anton',
        'anton',
        'beatrice',
        'felicia',
        'niklas',
        'adam',
        'adam',
        'beata',
        'beata',
    ]
    token2id: Token2Id = Token2Id().ingest(tokens).close()

    assert token2id.data == {'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4, 'beata': 5}
    assert dict(token2id.tf) == {0: 3, 1: 2, 2: 1, 3: 1, 4: 1, 5: 2}

    token2id_compressed = token2id.compress(tf_threshold=1, inplace=False)
    assert token2id_compressed is token2id
    assert token2id_compressed.fallback_token is None
    with pytest.raises(KeyError):
        _ = token2id_compressed["roger"]

    token2id_compressed = token2id.compress(tf_threshold=2, inplace=False)
    assert dict(token2id_compressed.data) == {'adam': 0, 'anton': 1, 'beata': 2, GLOBAL_TF_THRESHOLD_MASK_TOKEN: 3}
    assert dict(token2id_compressed.tf) == {0: 3, 1: 2, 2: 2, 3: 3}
    assert token2id.fallback_token is None
    assert token2id_compressed.fallback_token is not None
    assert token2id_compressed["roger"] == token2id_compressed[GLOBAL_TF_THRESHOLD_MASK_TOKEN]
    assert "roger" not in token2id_compressed

    with pytest.raises(ClosedVocabularyError):
        token2id_compressed["roger"] = 99

    assert token2id_compressed['word_that_dont_exists'] == token2id_compressed[GLOBAL_TF_THRESHOLD_MASK_TOKEN]

    token2id_compressed = token2id.compress(tf_threshold=2, keeps={4}, inplace=False)
    assert dict(token2id_compressed.data) == {
        'adam': 0,
        'anton': 1,
        'niklas': 2,
        'beata': 3,
        GLOBAL_TF_THRESHOLD_MASK_TOKEN: 4,
    }
    assert dict(token2id_compressed.tf) == {0: 3, 1: 2, 2: 1, 3: 2, 4: 2}

    token2id: Token2Id = Token2Id().ingest(tokens).ingest([GLOBAL_TF_THRESHOLD_MASK_TOKEN]).close()
    token2id_compressed = token2id.compress(tf_threshold=2, inplace=False)
    assert dict(token2id_compressed.data) == {
        'adam': 0,
        'anton': 1,
        'beata': 2,
        GLOBAL_TF_THRESHOLD_MASK_TOKEN: 3,
    }
    assert dict(token2id_compressed.tf) == {0: 3, 1: 2, 2: 2, 3: 4}

    token2id: Token2Id = Token2Id().ingest(tokens).ingest([GLOBAL_TF_THRESHOLD_MASK_TOKEN]).close()
    token2id.compress(tf_threshold=2, inplace=True)
    assert dict(token2id.data) == {
        'adam': 0,
        'anton': 1,
        'beata': 2,
        GLOBAL_TF_THRESHOLD_MASK_TOKEN: 3,
    }
    assert dict(token2id.tf) == {0: 3, 1: 2, 2: 2, 3: 4}
