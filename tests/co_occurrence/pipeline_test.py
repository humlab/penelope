import os

from penelope import pipeline
from penelope.co_occurrence import Bundle, ContextOpts
from penelope.corpus import TokenizedCorpus, VectorizedCorpus

from ..fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus

jj = os.path.join


def test_pipeline_to_co_occurrence_succeeds():

    tokenized_corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.tokenized_corpus()
    context_opts: ContextOpts = ContextOpts(context_width=2, concept={}, ignore_concept=False, ignore_padding=False)
    bundle: Bundle = (
        pipeline.CorpusPipeline(config=config)
        .load_corpus(tokenized_corpus)
        .vocabulary()
        .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=True)
        .to_corpus_co_occurrence(context_opts=context_opts, global_threshold_count=1)
        .single()
        .content
    )

    assert isinstance(bundle, Bundle)

    corpus: VectorizedCorpus = bundle.corpus

    assert isinstance(corpus, VectorizedCorpus)
    assert corpus.data.shape == (len(tokenized_corpus.document_index), len(corpus.token2id))
