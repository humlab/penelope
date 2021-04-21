import pytest
from penelope.pipeline.interfaces import Token2Id


def test_interfaces_token2id_get():

    token2id = Token2Id()

    token_id = token2id['apa']

    assert token_id == 0
    assert 'apa' in token2id


def test_interfaces_token2id_ingest():

    token2id = Token2Id()

    token2id.ingest(['apa', 'banan', 'soffa'])

    assert 'apa' in token2id
    assert 'banan' in token2id
    assert 'soffa' in token2id


def test_interfaces_token2id_close():

    token2id = Token2Id()

    token2id.ingest(['apa', 'banan', 'soffa'])
    token2id.close()

    with pytest.raises(KeyError):
        _ = token2id['hängmatta']

    token2id.open()
    _ = token2id['hängmatta']

    assert 'hängmatta' in token2id


def test_interfaces_token2id_reverse():

    token2id = Token2Id()

    id2token = token2id.ingest(['apa', 'banan', 'soffa']).id2token()
    assert id2token[0] == 'apa'
    assert id2token[1] == 'banan'
    assert id2token[2] == 'soffa'
