import os

import pytest
from penelope.co_occurrence import Bundle, ContextOpts
from penelope.corpus import ClosedVocabularyError, TokenizedCorpus, VectorizedCorpus
from penelope.pipeline import CorpusConfig, CorpusPipeline

from ..fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus

jj = os.path.join

CONTEXT_OPTS: ContextOpts = ContextOpts(context_width=2, concept={}, ignore_concept=False, ignore_padding=False)


def test_pipeline_to_co_occurrence_ingest_prohobited_if_vocabulary_exists():

    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: CorpusConfig = CorpusConfig.tokenized_corpus()

    ingest_tokens = True
    with pytest.raises(ClosedVocabularyError):
        _: Bundle = (
            CorpusPipeline(config=config)
            .load_corpus(tokenized_corpus)
            .vocabulary(lemmatize=False)
            .to_document_co_occurrence(context_opts=CONTEXT_OPTS, ingest_tokens=ingest_tokens)
            .to_corpus_co_occurrence(context_opts=CONTEXT_OPTS, global_threshold_count=1)
            .single()
            .content
        )


def test_pipeline_to_co_occurrence_can_create_new_vocabulary():

    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: CorpusConfig = CorpusConfig.tokenized_corpus()

    ingest_tokens = False
    bundle: Bundle = (
        CorpusPipeline(config=config)
        .load_corpus(tokenized_corpus)
        .vocabulary(lemmatize=False)
        .to_document_co_occurrence(context_opts=CONTEXT_OPTS, ingest_tokens=ingest_tokens)
        .to_corpus_co_occurrence(context_opts=CONTEXT_OPTS, global_threshold_count=1)
        .single()
        .content
    )
    assert isinstance(bundle, Bundle)
    corpus: VectorizedCorpus = bundle.corpus
    assert isinstance(corpus, VectorizedCorpus)


def test_pipeline_to_co_occurrence_can_ingest_new_vocabulary():
    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: CorpusConfig = CorpusConfig.tokenized_corpus()

    ingest_tokens = True
    bundle: Bundle = (
        CorpusPipeline(config=config)
        .load_corpus(tokenized_corpus)
        .to_document_co_occurrence(context_opts=CONTEXT_OPTS, ingest_tokens=ingest_tokens)
        .to_corpus_co_occurrence(context_opts=CONTEXT_OPTS, global_threshold_count=1)
        .single()
        .content
    )
    assert isinstance(bundle, Bundle)
    corpus: VectorizedCorpus = bundle.corpus
    assert isinstance(corpus, VectorizedCorpus)
