import os
import pathlib
import uuid
from typing import List, Tuple, Union

from penelope.co_occurrence import Bundle, ContextOpts, CoOccurrenceHelper
from penelope.corpus import TokenizedCorpus
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
        corpus_config=corpus_config,
        corpus_filename=source_filename,
        enable_checkpoint=True,
        force_checkpoint=False,
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
    bundle: Bundle = create_simple_bundle_by_pipeline(
        data=simple_corpus,
        context_opts=context_opts,
        folder=folder,
        tag=tag,
    )
    return bundle


def create_simple_bundle_by_pipeline(
    data: Union[TokenizedCorpus, List[Tuple[str, List[str]]]],
    context_opts: ContextOpts,
    folder: str = OUTPUT_FOLDER,
    tag: str = "TERRA",
):
    if not isinstance(data, TokenizedCorpus):
        data: TokenizedCorpus = very_simple_corpus(data)
    config: CorpusConfig = CorpusConfig.tokenized_corpus_config()
    bundle: Bundle = (
        CorpusPipeline(config=config)
        .load_corpus(data)
        .vocabulary(lemmatize=False)
        .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=False)
        .to_corpus_co_occurrence(context_opts=context_opts, global_threshold_count=1)
        .single()
        .content
    )
    bundle.folder = folder
    bundle.tag = tag
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


def fake_config() -> CorpusConfig:
    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    return corpus_config
