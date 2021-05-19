from penelope.corpus import Token2Id


def test_token2id_find():

    token2id: Token2Id = Token2Id({'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4})

    assert set(token2id.find(what='adam')) == set([0])
    assert set(token2id.find(what='a*')) == set([0, 1])
    assert set(token2id.find(what=['a*', 'f*'])) == set([0, 1, 3])
    assert set(token2id.find(what=['a*', 'beatrice'])) == set([0, 1, 2])

def test_token2id_ingest():

    tokens = ['adam', 'anton', 'beatrice', 'felicia', 'niklas']
    token2id: Token2Id = Token2Id().ingest(tokens)

    assert token2id.data == {'adam': 0, 'anton': 1, 'beatrice': 2, 'felicia': 3, 'niklas': 4}
