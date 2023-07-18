import pandas as pd
import pytest

from penelope import corpus as pc
from penelope import utility
from penelope.corpus.serialize import SerializeOpts
from penelope.pipeline import PoSCountMixIn, TokenCountMixIn, sparv

from ..fixtures import TEST_CSV_POS_DOCUMENT

# pylint: disable=redefined-outer-name


@pytest.fixture
def document_index() -> pd.DataFrame:
    di_str: str = """
filename;year;year_id;document_name;document_id;title;n_raw_tokens
A.txt;2019;1;A;0;Even break;68
B.txt;2019;2;B;1;Night;59
"""
    di: pd.DataFrame = pc.load_document_index_from_str(di_str, sep=';')
    return di


@pytest.fixture
def tagged_frame() -> pd.DataFrame:
    tf: pd.DataFrame = sparv.SparvCsvSerializer().deserialize(
        content=TEST_CSV_POS_DOCUMENT,
        options=SerializeOpts(text_column='token', lemma_column='baseform', pos_column='pos'),
    )
    return tf


@pytest.fixture
def pos_schema() -> utility.PoS_Tag_Scheme:
    return utility.PoS_Tag_Schemes.SUC


def test_PoSCountMixIn(document_index: pd.DataFrame, tagged_frame: pd.DataFrame, pos_schema: utility.PoS_Tag_Scheme):
    task = PoSCountMixIn()

    assert len(task.document_tfs) == 0

    task.register_pos_counts2(document_name='A', tagged_frame=tagged_frame, pos_schema=pos_schema)

    assert task.document_tfs == {
        'A': {
            'Adjective': 2,
            'Adverb': 2,
            'Conjunction': 0,
            'Delimiter': 3,
            'Noun': 7,
            'Numeral': 1,
            'Other': 1,
            'Preposition': 1,
            'Pronoun': 3,
            'Verb': 3,
            'n_raw_tokens': 20,
            'n_tokens': 20,
            'document_name': 'A',
        }
    }

    assert 'Adjective' not in document_index.columns

    task.flush_pos_counts2(document_index=document_index)

    assert 'Adjective' in document_index.columns


def test_PoSTokenMixIn(document_index: pd.DataFrame):
    task = TokenCountMixIn()

    assert len(task.token_counts) == 0

    task.register_token_count(document_name='A', n_tokens=100)

    assert task.token_counts == {'A': ('A', 100)}

    assert 'n_tokens' not in document_index.columns

    task.flush_token_counts(document_index=document_index)

    assert 'n_tokens' in document_index.columns

    assert document_index.n_tokens['A'] == 100
    assert document_index.n_tokens['B'] == 0
