import os
from typing import Set

import pandas as pd
import pytest
from penelope.co_occurrence import Bundle, CoOccurrenceHelper, to_filename

jj = os.path.join

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = to_filename(folder=folder, tag=tag)

    bundle: Bundle = Bundle.load(filename, compute_frame=False)

    return bundle


@pytest.fixture(scope="module")
def helper(bundle: Bundle) -> CoOccurrenceHelper:
    helper: CoOccurrenceHelper = CoOccurrenceHelper(
        corpus=bundle.corpus,
        source_token2id=bundle.token2id,
        pivot_keys=None,
    )
    return helper


def test_co_occurrence_helper_reset(helper: CoOccurrenceHelper):

    helper.reset()

    assert (helper.data == helper.co_occurrences).all().all()


def test_co_occurrence_helper_decode(helper: CoOccurrenceHelper):

    helper.co_occurrences.drop(columns=["w1", "w2", "token"], inplace=True, errors="ignore")

    helper.decode()

    assert all(x in helper.co_occurrences.columns for x in ["w1", "w2", "token"])


def test_co_occurrence_helper_groupby(helper: CoOccurrenceHelper):

    helper.reset()

    assert 'document_id' in helper.value.columns
    assert 'year' not in helper.value.columns

    helper.groupby('year')

    assert 'document_id' not in helper.value.columns
    assert 'year' in helper.value.columns


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

    helper.reset().groupby('year').match("nations").decode()
    assert all(helper.value.token.str.contains("nations"))

    helper.reset().groupby('year').match("nat*").decode()
    assert all(helper.value.token.str.contains("nat"))


def test_co_occurrence_helper_exclude(helper: CoOccurrenceHelper):
    helper.reset().groupby('year').exclude("united").decode()
    assert all(~helper.value.token.str.contains("united"))


def test_co_occurrence_helper_rank(helper: CoOccurrenceHelper):
    top_pairs = helper.reset().groupby('year').decode().rank(10).value.token.tolist()
    assert 'constitution/united' in top_pairs


def test_co_occurrence_helper_largest(helper: CoOccurrenceHelper):

    expected: Set[str] = set(
        helper.data[helper.data.year == 1997].sort_values('value', ascending=False).head(3)['token'].tolist()
    )
    data: pd.DataFrame = helper.reset().groupby('year').decode().largest(3).value
    largest = set(data[data.year == 1997].token.tolist())
    assert largest == expected


def test_co_occurrence_helper_head(helper: CoOccurrenceHelper):
    heads = helper.reset().groupby('year').decode().head(10).value.token.tolist()
    assert len(heads) == 10
