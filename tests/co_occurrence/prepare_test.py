import os

from penelope import co_occurrence
from penelope.co_occurrence import Bundle, CoOccurrenceHelper

from ..fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS, very_simple_corpus
from ..utils import OUTPUT_FOLDER
from .utils import create_co_occurrence_bundle

jj = os.path.join


def create_simple_bundle() -> Bundle:
    tag: str = "TERRA"
    folder: str = jj(OUTPUT_FOLDER, tag)
    simple_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        concept={}, ignore_concept=False, context_width=2
    )
    bundle: Bundle = create_co_occurrence_bundle(
        corpus=simple_corpus, context_opts=context_opts, folder=folder, tag=tag
    )
    return bundle


def create_bundle_helper(bundle: Bundle) -> CoOccurrenceHelper:
    helper: CoOccurrenceHelper = CoOccurrenceHelper(
        bundle.co_occurrences,
        bundle.token2id,
        bundle.document_index,
    )
    return helper


def create_simple_helper() -> CoOccurrenceHelper:
    return create_bundle_helper(
        create_bundle_helper(create_simple_bundle()),
    )


def test_co_occurrence_helper_reset():

    helper: CoOccurrenceHelper = create_simple_helper()

    helper.reset()

    assert (helper.data == helper.co_occurrences).all().all()


def test_co_occurrence_groupby():

    helper: CoOccurrenceHelper = create_simple_helper()

    helper.reset()

    yearly_co_occurrences = helper.groupby('year').value

    assert yearly_co_occurrences is not None
