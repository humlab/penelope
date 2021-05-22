import os

from penelope import pipeline
from penelope.corpus import TokenizedCorpus, VectorizedCorpus

from ..fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus

jj = os.path.join


def test_pipeline_to_co_occurrence_succeeds():

    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.tokenized_corpus()
    corpus: VectorizedCorpus = (
        pipeline.CorpusPipeline(config=config)
        .load_corpus(tokenized_corpus)
        .vocabulary()
        .to_document_co_occurrence()
        .to_corpus_co_occurrence()
        .single()
        .content
    )

    assert isinstance(corpus, VectorizedCorpus)
    assert corpus.data.shape[0] == 5
    assert len(corpus.token2id) == corpus.data.shape[1]
