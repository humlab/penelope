import os

import penelope.co_occurrence as co_occurrence
import penelope.workflows as workflows
import pytest
from penelope.co_occurrence.partitioned import ComputeResult
from penelope.corpus import TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility import PoS_Tag_Scheme, PoS_Tag_Schemes, PropertyValueMaskingOpts, pos_tags_to_str

from ..fixtures import FakeComputeOptsSpacyCSV

# pylint: disable=redefined-outer-name


def fake_config() -> CorpusConfig:
    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/ssi_corpus_config.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    return corpus_config


@pytest.fixture(scope='module')
def config():
    return fake_config()


def test_spaCy_co_occurrence_pipeline(config):

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/test_data/legal_instrument_five_docs_test_pos_csv.zip"
    target_filename = './tests/output/SSI-co-occurrence-JJVBNN-window-9.csv'
    if os.path.isfile(target_filename):
        os.remove(target_filename)

    # .folder(folder='./tests/test_data')
    pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
    tokens_transform_opts: TokensTransformOpts = TokensTransformOpts()
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes=pos_tags_to_str(pos_scheme.Adjective + pos_scheme.Verb + pos_scheme.Noun),
        # FIXME: This test will fail:
        pos_paddings=pos_tags_to_str(pos_scheme.Conjunction)
    )
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(
        is_punct=False,
    )
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(context_width=4)
    global_threshold_count: int = 1
    partition_column: str = 'year'

    compute_result: ComputeResult = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_filename=config.pipeline_payload.source,
        tokens_transform_opts=tokens_transform_opts,
        extract_tagged_tokens_opts=extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=tagged_tokens_filter_opts,
        context_opts=context_opts,
        global_threshold_count=global_threshold_count,
        partition_column=partition_column,
        checkpoint_filename=checkpoint_filename,
    ).value()

    compute_result.co_occurrences.to_csv(target_filename, sep='\t')

    assert os.path.isfile(target_filename)

    os.remove(target_filename)


def test_spaCy_co_occurrence_workflow(config):

    partition_key: str = 'year'
    args = FakeComputeOptsSpacyCSV(
        corpus_tag="VENUS",
        corpus_filename=config.pipeline_payload.source,
    )
    args.context_opts = co_occurrence.ContextOpts(
        context_width=4, ignore_concept=True
    )  # , concept={''}, ignore_concept=True)

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/output/co_occurrence_test_pos_csv.zip"

    compute_result: co_occurrence.ComputeResult = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_filename=None,
        tokens_transform_opts=args.tokens_transform_opts,
        extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
        context_opts=args.context_opts,
        global_threshold_count=args.count_threshold,
        partition_column=partition_key,
        checkpoint_filename=checkpoint_filename,
    ).value()

    assert compute_result.co_occurrences is not None
    assert compute_result.document_index is not None
    assert len(compute_result.co_occurrences) > 0

    co_occurrence_filename = co_occurrence.folder_and_tag_to_filename(folder=args.target_folder, tag=args.corpus_tag)

    corpus: VectorizedCorpus = co_occurrence.to_vectorized_corpus(
        co_occurrences=compute_result.co_occurrences, document_index=compute_result.document_index, value_column='value'
    )

    bundle = co_occurrence.Bundle(
        corpus=corpus,
        corpus_tag=args.corpus_tag,
        corpus_folder=args.target_folder,
        co_occurrences=compute_result.co_occurrences,
        document_index=compute_result.document_index,
        compute_options=co_occurrence.create_options_bundle(
            reader_opts=config.text_reader_opts,
            tokens_transform_opts=args.tokens_transform_opts,
            context_opts=args.context_opts,
            extract_tokens_opts=args.extract_tagged_tokens_opts,
            input_filename=args.corpus_filename,
            output_filename=co_occurrence_filename,
            partition_keys=[partition_key],
            count_threshold=args.count_threshold,
        ),
    )

    co_occurrence.store_bundle(co_occurrence_filename, bundle)


def test_spaCy_co_occurrence_pipeline3(config):
    corpus_filename = './tests/test_data/legal_instrument_five_docs_test.zip'
    args = FakeComputeOptsSpacyCSV(
        corpus_filename=corpus_filename,
        corpus_tag="SATURNUS",
    )

    workflows.co_occurrence.compute(
        args=args,
        corpus_config=config,
        checkpoint_file='./tests/output/co_occurrence_checkpoint_pos.csv.zip',
    )
