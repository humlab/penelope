import os

import penelope.co_occurrence as co_occurrence
import penelope.workflows as workflows
import pytest
from penelope.corpus import TokensTransformOpts
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


@pytest.mark.long_running
def test_spaCy_co_occurrence_pipeline(config):

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/test_data/legal_instrument_five_docs_test_pos_csv.zip"
    target_filename = './tests/output/SSI-co-occurrence-JJVBNN-window-9.csv'
    if os.path.isfile(target_filename):
        os.remove(target_filename)

    # .folder(folder='./tests/test_data')
    pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
    transform_opts: TokensTransformOpts = TokensTransformOpts()
    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes=pos_tags_to_str(pos_scheme.Adjective + pos_scheme.Verb + pos_scheme.Noun),
        pos_paddings=pos_tags_to_str(pos_scheme.Conjunction),
    )
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(
        is_punct=False,
    )
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        context_width=4,
        partition_keys=['document_id'],
    )
    global_threshold_count: int = 1

    value: co_occurrence.Bundle = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_filename=config.pipeline_payload.source,
        transform_opts=transform_opts,
        context_opts=context_opts,
        extract_opts=extract_opts,
        filter_opts=filter_opts,
        global_threshold_count=global_threshold_count,
        checkpoint_filename=checkpoint_filename,
    ).value()

    value.co_occurrences.to_csv(target_filename, sep='\t')

    assert os.path.isfile(target_filename)

    os.remove(target_filename)


@pytest.mark.long_running
def test_spaCy_co_occurrence_workflow():
    """Note: Use the output from this test case to update the tests/test_data/VENUS test data VENUS-TESTDATA"""

    config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')

    config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    args = FakeComputeOptsSpacyCSV(
        corpus_tag="VENUS",
        corpus_filename=config.pipeline_payload.source,
    )
    args.context_opts = co_occurrence.ContextOpts(context_width=4, ignore_concept=True, partition_keys=['document_id'])

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/output/co_occurrence_test_pos_csv.zip"

    bundle: co_occurrence.Bundle = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_filename=None,
        transform_opts=args.transform_opts,
        extract_opts=args.extract_opts,
        filter_opts=args.filter_opts,
        context_opts=args.context_opts,
        global_threshold_count=args.tf_threshold,
        checkpoint_filename=checkpoint_filename,
    ).value()

    assert bundle.corpus is not None
    assert bundle.token2id is not None
    assert bundle.document_index is not None

    bundle.tag = args.corpus_tag
    bundle.folder = args.target_folder
    bundle.co_occurrences = bundle.corpus.to_co_occurrences(bundle.token2id)

    bundle.store()


@pytest.mark.long_running
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
