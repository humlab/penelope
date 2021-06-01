import os

from penelope.co_occurrence import Bundle, ContextOpts
from penelope.corpus import VectorizedCorpus
from tests.co_occurrence.utils import create_simple_bundle_by_pipeline

from ..fixtures import SIMPLE_CORPUS_ABCDE_5DOCS

jj = os.path.join


def test_pipeline_to_co_occurrence_succeeds():

    context_opts: ContextOpts = ContextOpts(context_width=2, concept={}, ignore_concept=False, ignore_padding=False)
    bundle: create_simple_bundle_by_pipeline(SIMPLE_CORPUS_ABCDE_5DOCS, context_opts)

    assert isinstance(bundle, Bundle)

    corpus: VectorizedCorpus = bundle.corpus

    assert isinstance(corpus, VectorizedCorpus)
