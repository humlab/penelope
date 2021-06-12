import os
import pathlib
import uuid
from typing import List, Tuple

from penelope.co_occurrence import Bundle, ContextOpts, CoOccurrenceHelper
from penelope.co_occurrence.utility import compute_non_partitioned_corpus_co_occurrence
from penelope.corpus import ITokenizedCorpus, Token2Id, TokenizedCorpus, VectorizedCorpus
from penelope.pipeline import CorpusConfig, CorpusPipeline, sparv

from ..fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS, very_simple_corpus
from ..utils import OUTPUT_FOLDER

jj = os.path.join


def create_tranströmer_to_tagged_frame_pipeline() -> CorpusPipeline:

    config_filename = './tests/test_data/tranströmer.yml'
    source_filename = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip'
    checkpoint_filename = f'./tests/output/{uuid.uuid1()}.checkpoint.zip'

    corpus_config: CorpusConfig = CorpusConfig.load(config_filename)

    corpus_config.pipeline_payload.source = source_filename
    corpus_config.pipeline_payload.document_index_source = None

    pathlib.Path(checkpoint_filename).unlink(missing_ok=True)

    p: CorpusPipeline = sparv.to_tagged_frame_pipeline(
        corpus_config,
        corpus_filename=source_filename,
    )  # .checkpoint(checkpoint_filename)

    return p


def create_very_simple_tokens_pipeline(data: List[Tuple[str, List[str]]]) -> CorpusPipeline:
    corpus: TokenizedCorpus = very_simple_corpus(data)
    p: CorpusPipeline = CorpusPipeline(config=None).load_corpus(corpus)
    return p


def create_simple_bundle() -> Bundle:
    tag: str = "TERRA"
    folder: str = jj(OUTPUT_FOLDER, tag)
    simple_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    context_opts: ContextOpts = ContextOpts(concept=set(), ignore_concept=False, context_width=2)
    bundle: Bundle = create_co_occurrence_bundle(
        corpus=simple_corpus, context_opts=context_opts, folder=folder, tag=tag
    )
    return bundle


def create_simple_bundle_by_pipeline(data: List[Tuple[str, List[str]]], context_opts: ContextOpts):
    tokenized_corpus: TokenizedCorpus = very_simple_corpus(data)
    config: CorpusConfig = CorpusConfig.tokenized_corpus()
    bundle: Bundle = (
        CorpusPipeline(config=config)
        .load_corpus(tokenized_corpus)
        .vocabulary(lemmatize=True)
        .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=True)
        .to_corpus_co_occurrence(context_opts=context_opts, global_threshold_count=1)
        .single()
        .content
    )
    return bundle


def create_bundle_helper(bundle: Bundle) -> CoOccurrenceHelper:
    helper: CoOccurrenceHelper = CoOccurrenceHelper(
        corpus=bundle.corpus,
        source_token2id=bundle.token2id,
        pivot_keys=None,
    )
    return helper


def create_simple_helper() -> CoOccurrenceHelper:
    return create_bundle_helper(
        create_bundle_helper(create_simple_bundle()),
    )


def create_co_occurrence_bundle(
    *, corpus: ITokenizedCorpus, context_opts: ContextOpts, folder: str, tag: str
) -> Bundle:

    token2id: Token2Id = Token2Id(corpus.token2id)

    bundle: Bundle = compute_non_partitioned_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=token2id,
        context_opts=context_opts,
        global_threshold_count=1,
    )

    corpus: VectorizedCorpus = VectorizedCorpus.from_co_occurrences(
        co_occurrences=bundle.co_occurrences,
        document_index=bundle.document_index,
        token2id=token2id,
    )

    bundle.corpus = corpus
    bundle.tag = tag
    bundle.folder = folder

    return bundle


def fake_config() -> CorpusConfig:
    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    return corpus_config
