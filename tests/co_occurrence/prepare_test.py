import os

import pandas as pd
import pytest
from penelope.co_occurrence import Bundle, CoOccurrenceHelper, to_filename
from penelope.co_occurrence.prepare import decode_tokens
from penelope.corpus.dtm.ttm import CoOccurrenceVocabularyHelper, to_word_pair_token
from penelope.corpus.dtm.vectorized_corpus import VectorizedCorpus

jj = os.path.join

# pylint: disable=redefined-outer-name


def create_bundle() -> Bundle:
    folder, tag = './tests/test_data/VENUS', 'VENUS'
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle


def create_helper(bundle: Bundle, period_specifier: str, pivot_key: str):
    corpus: VectorizedCorpus = bundle.corpus.group_by_time_period_optimized(
        time_period_specifier=period_specifier,
        target_column_name=pivot_key,
    )

    co_occurrences = corpus.to_co_occurrences(source_token2id=bundle.token2id, partition_key=pivot_key)

    helper: CoOccurrenceHelper = CoOccurrenceHelper(
        corpus=bundle.corpus,
        source_token2id=bundle.token2id,
        pivot_keys=[pivot_key],
        co_occurrences=co_occurrences,
    )
    return helper


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    return create_bundle()


@pytest.fixture(scope="module")
def helper(bundle: Bundle) -> CoOccurrenceHelper:
    helper: CoOccurrenceHelper = create_helper(bundle, "year", "time_period")
    return helper


def test_co_occurrence_helper_reset(helper: CoOccurrenceHelper):

    helper.reset()

    assert (helper.data == helper.co_occurrences).all().all()


def test_co_occurrence_helper_decode(helper: CoOccurrenceHelper):

    helper.co_occurrences.drop(columns=["w1", "w2", "token"], inplace=True, errors="ignore")

    helper.decode()

    assert all(x in helper.data.columns for x in ["w1", "w2", "token"])


def test_co_occurrence_helper_trunk_by_global_count(helper: CoOccurrenceHelper):

    helper.reset()

    before_count = len(helper.value)

    expected_removals = set(
        helper.value[helper.value.groupby('token')['value'].transform('sum') >= 10].token.unique().tolist()
    )

    helper = helper.trunk_by_global_count(threshold=10)

    after_count = len(helper.value)

    assert before_count > after_count
    assert len(helper.value[helper.value.value.isin(expected_removals)]) == 0


def test_co_occurrence_helper_match(helper: CoOccurrenceHelper):

    helper.reset().match("nations").decode()
    assert all(helper.value.token.str.contains("nations"))

    helper.reset().match("nat*").decode()
    assert all(helper.value.token.str.contains("nat"))


def test_co_occurrence_helper_exclude(helper: CoOccurrenceHelper):
    helper.reset().exclude("united").decode()
    assert all(~helper.value.token.str.contains("united"))


def test_co_occurrence_helper_rank(helper: CoOccurrenceHelper):
    top_pairs = helper.reset().decode().rank(10).value.token.tolist()
    assert 'constitution/united' in top_pairs


def test_co_occurrence_helper_largest(helper: CoOccurrenceHelper):

    data: pd.DataFrame = helper.reset().largest(3).decode().value
    largest = set(data[data.time_period == 1997].token.tolist())
    assert largest == {'article/generation', 'ensure/generation', 'should/generation'}


def test_co_occurrence_helper_head(helper: CoOccurrenceHelper):
    heads = helper.reset().decode().head(10).value.token.tolist()
    assert len(heads) == 10


def test_create_co_occurrence_vocabulary():

    bundle: Bundle = create_bundle()
    co_occurrences: pd.DataFrame = bundle.co_occurrences
    co_occurrences = decode_tokens(co_occurrences, bundle.token2id.id2token, bundle.corpus.id2token)
    vocab, vocab_mapping = CoOccurrenceVocabularyHelper.create_co_occurrence_vocabulary(co_occurrences, bundle.token2id)

    id2token = {v: k for k, v in vocab.items()}

    fg = bundle.token2id.id2token.get
    assert all(to_word_pair_token(k[0], k[1], fg) == id2token[v] for k, v in vocab_mapping.items())

    tokens: pd.Series = (
        co_occurrences[['w1_id', 'w2_id']].apply(lambda x: vocab_mapping.get((x[0], x[1])), axis=1).apply(id2token.get)
    )

    assert vocab is not None
    assert len(vocab) == len(vocab_mapping)
    assert all(tokens == co_occurrences.token)
