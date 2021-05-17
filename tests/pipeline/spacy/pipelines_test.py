import os

import penelope.co_occurrence as co_occurrence
import penelope.co_occurrence.partition_by_document as co_occurrence_module
import penelope.workflows as workflows
import pytest
from penelope.co_occurrence import CoOccurrenceComputeResult
from penelope.corpus import TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility import PoS_Tag_Scheme, PoS_Tag_Schemes, PropertyValueMaskingOpts, pos_tags_to_str

from ..fixtures import FakeComputeOptsSpacyCSV

# pylint: disable=redefined-outer-name


def fake_config() -> CorpusConfig:
    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')

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
        pos_paddings=pos_tags_to_str(pos_scheme.Conjunction),
    )
    tagged_tokens_filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(
        is_punct=False,
    )
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        context_width=4,
        partition_keys=['document_id'],
    )
    global_threshold_count: int = 1

    value: CoOccurrenceComputeResult = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_filename=config.pipeline_payload.source,
        tokens_transform_opts=tokens_transform_opts,
        context_opts=context_opts,
        extract_tagged_tokens_opts=extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=tagged_tokens_filter_opts,
        global_threshold_count=global_threshold_count,
        checkpoint_filename=checkpoint_filename,
    ).value()

    value.co_occurrences.to_csv(target_filename, sep='\t')

    assert os.path.isfile(target_filename)

    os.remove(target_filename)


def test_spaCy_co_occurrence_workflow(config):

    args = FakeComputeOptsSpacyCSV(
        corpus_tag="VENUS",
        corpus_filename=config.pipeline_payload.source,
    )
    args.context_opts = co_occurrence.ContextOpts(
        context_width=4, ignore_concept=True, partition_keys=['document_id']
    )  # , concept={''}, ignore_concept=True)

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/output/co_occurrence_test_pos_csv.zip"

    value: co_occurrence.CoOccurrenceComputeResult = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_filename=None,
        tokens_transform_opts=args.tokens_transform_opts,
        extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
        context_opts=args.context_opts,
        global_threshold_count=args.count_threshold,
        checkpoint_filename=checkpoint_filename,
    ).value()

    assert value.co_occurrences is not None
    assert value.document_index is not None
    assert len(value.co_occurrences) > 0

    corpus: VectorizedCorpus = co_occurrence_module.co_occurrence_dataframe_to_vectorized_corpus(
        co_occurrences=value.co_occurrences,
        token2id=value.token2id,
        document_index=value.document_index,
        # partition_key=args.context_opts.partition_keys[0],
    )

    bundle = co_occurrence.Bundle(
        corpus=corpus,
        tag=args.corpus_tag,
        folder=args.target_folder,
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        compute_options=co_occurrence.create_options_bundle(
            reader_opts=config.text_reader_opts,
            tokens_transform_opts=args.tokens_transform_opts,
            context_opts=args.context_opts,
            extract_tokens_opts=args.extract_tagged_tokens_opts,
            input_filename=args.corpus_filename,
            output_filename=co_occurrence.to_filename(folder=args.target_folder, tag=args.corpus_tag),
            count_threshold=args.count_threshold,
        ),
    )

    bundle.store()


def test_spaCy_co_occurrence_pipeline3(config):
    corpus_filename = './tests/test_data/legal_instrument_five_docs_test.zip'
    args = FakeComputeOptsSpacyCSV(
        corpus_filename=corpus_filename,
        corpus_tag="SATURNUS",
    )

    workflows.co_occurrence.compute_partitioned_by_key(
        args=args,
        corpus_config=config,
        checkpoint_file='./tests/output/co_occurrence_checkpoint_pos.csv.zip',
    )
