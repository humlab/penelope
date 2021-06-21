import os
import pathlib
import re
from typing import List

import pandas as pd
import penelope.workflows as workflows
import pytest
from penelope.corpus import TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextTransformOpts
from penelope.pipeline import CorpusConfig, CorpusPipeline, DocumentPayload
from penelope.pipeline.config import create_pipeline_factory
from penelope.pipeline.tasks import Checkpoint, Tqdm
from penelope.utility import PropertyValueMaskingOpts
from sklearn.feature_extraction.text import CountVectorizer
from tests.utils import OUTPUT_FOLDER

from .fixtures import ComputeOptsSpacyCSV

CORPUS_FOLDER = './tests/test_data'

# pylint: disable=redefined-outer-name


def fake_config() -> CorpusConfig:

    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    return corpus_config


@pytest.fixture(scope='module')
def config():
    return fake_config()


def test_corpus_config_set_folder(config: CorpusConfig):

    current_source = config.pipeline_payload.source
    config.pipeline_payload.folders(CORPUS_FOLDER)

    assert config.pipeline_payload.source == current_source


def test_load_text_returns_payload_with_expected_document_index(config: CorpusConfig):

    transform_opts = TextTransformOpts()

    pipeline = CorpusPipeline(config=config).load_text(
        reader_opts=config.text_reader_opts, transform_opts=transform_opts
    )
    assert pipeline is not None

    payloads: List[DocumentPayload] = [x for x in pipeline.resolve()]

    assert len(payloads) == 5
    assert len(pipeline.payload.document_index) == 5
    assert len(pipeline.payload.metadata) == 5
    assert pipeline.payload.pos_schema_name == "Universal"
    assert pipeline.payload.get('text_reader_opts') == config.text_reader_opts.props
    assert isinstance(pipeline.payload.document_index, pd.DataFrame)

    columns = pipeline.payload.document_index.columns.tolist()
    assert columns == [
        'section_id',
        'unesco_id',
        'filename',
        'type',
        'href',
        'year',
        'date',
        'city',
        'x.title',
        'document_id',
        'document_name',
    ]
    assert all(x.split(':')[0] in columns for x in config.text_reader_opts.filename_fields)
    assert pipeline.payload.document_lookup('RECOMMENDATION_0201_049455_2017.txt')['unesco_id'] == 49455


@pytest.mark.long_running
def test_pipeline_load_text_tag_checkpoint_stores_checkpoint(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(OUTPUT_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    transform_opts = TextTransformOpts()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pathlib.Path(checkpoint_filename).unlink(missing_ok=True)

    _ = (
        CorpusPipeline(config=config)
        .set_spacy_model(config.pipeline_payload.memory_store['spacy_model'])
        .load_text(reader_opts=config.text_reader_opts, transform_opts=transform_opts)
        .text_to_spacy()
        .tqdm()
        .spacy_to_pos_tagged_frame()
        .checkpoint(checkpoint_filename, force_checkpoint=False)
    ).exhaust()

    assert os.path.isfile(checkpoint_filename)
    pathlib.Path(checkpoint_filename).unlink(missing_ok=True)


def test_pipeline_can_load_pos_tagged_checkpoint(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    pipeline = CorpusPipeline(config=config).checkpoint(checkpoint_filename, force_checkpoint=False)

    payloads: List[DocumentPayload] = pipeline.to_list()

    assert len(payloads) == 5
    assert len(pipeline.payload.document_index) == 5
    assert isinstance(pipeline.payload.document_index, pd.DataFrame)


@pytest.mark.long_running
def test_pipeline_tagged_frame_to_tokens_succeeds(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|NOUN|', pos_paddings=None
    )
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)

    tagged_payload = next(
        CorpusPipeline(config=config).checkpoint(checkpoint_filename, force_checkpoint=False).resolve()
    )

    tokens_payload = next(
        CorpusPipeline(config=config)
        .checkpoint(checkpoint_filename, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=None)
        .resolve()
    )

    assert tagged_payload.filename == tokens_payload.filename
    assert len(tagged_payload.content[tagged_payload.content.pos_ == 'NOUN']) == len(tokens_payload.content)


@pytest.mark.long_running
def test_pipeline_tagged_frame_to_vocabulary_succeeds(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    pipeline: CorpusPipeline = (
        CorpusPipeline(config=config)
        .checkpoint(checkpoint_filename, force_checkpoint=False)
        .vocabulary(lemmatize=True, progress=False)
        .exhaust()
    )

    assert pipeline.payload.token2id is not None
    assert pipeline.payload.token2id.tf is not None
    assert len(pipeline.payload.token2id) == 1147
    assert len(pipeline.payload.token2id) == len(pipeline.payload.token2id.tf) is not None
    assert set(pipeline.payload.token2id.data.keys()) == {x.lower() for x in pipeline.payload.token2id.keys()}
    assert 'Cultural' not in pipeline.payload.token2id
    assert 'wars' not in pipeline.payload.token2id
    assert 'war' in pipeline.payload.token2id

    pipeline: CorpusPipeline = (
        CorpusPipeline(config=config)
        .checkpoint(checkpoint_filename, force_checkpoint=False)
        .vocabulary(lemmatize=False, progress=False)
        .exhaust()
    )

    assert len(pipeline.payload.token2id) == 1478
    assert 'Cultural' in pipeline.payload.token2id
    assert 'wars' in pipeline.payload.token2id

    assert pipeline.payload.token2id.tf[pipeline.payload.token2id['the']] == 704


@pytest.mark.long_running
def test_pipeline_tagged_frame_to_text_succeeds(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|NOUN|', pos_paddings=None
    )
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)

    tagged_payload = next(
        CorpusPipeline(config=config).checkpoint(checkpoint_filename, force_checkpoint=False).resolve()
    )

    text_payload = next(
        CorpusPipeline(config=config)
        .checkpoint(checkpoint_filename, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=None)
        .tokens_to_text()
        .resolve()
    )

    assert tagged_payload.filename == text_payload.filename
    assert len(tagged_payload.content[tagged_payload.content.pos_ == 'NOUN']) == len(text_payload.content.split())


def test_pipeline_take_succeeds(config: CorpusConfig):
    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(lemmatize=True)
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)

    take_payloads = (
        CorpusPipeline(config=config)
        .checkpoint(checkpoint_filename, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=None)
        .tokens_to_text()
        .take(2)
    )

    assert len(take_payloads) == 2


def test_pipeline_tagged_frame_to_tuple_succeeds(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|NOUN|', pos_paddings='|VERB|'
    )
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)

    payloads = (
        CorpusPipeline(config=config)
        .checkpoint(checkpoint_filename, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=None)
        .tokens_to_text()
        .to_document_content_tuple()
        .to_list()
    )
    assert len(payloads) == 5

    assert all(isinstance(payload.content, tuple) for payload in payloads)
    assert all(isinstance(payload.content[0], str) for payload in payloads)
    assert all(isinstance(payload.content[1], str) for payload in payloads)


def test_pipeline_find_task(config: CorpusConfig):
    p: CorpusPipeline = CorpusPipeline(config=config).checkpoint("dummy_name", force_checkpoint=False).tqdm()
    assert isinstance(p.find(Checkpoint), Checkpoint)
    assert isinstance(p.find(Tqdm), Tqdm)


def test_pipeline_to_dtm_succeeds(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|NOUN|', pos_paddings=None
    )
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)

    corpus: VectorizedCorpus = (
        (
            CorpusPipeline(config=config)
            .checkpoint(checkpoint_filename, force_checkpoint=False)
            .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=None)
            .tokens_transform(transform_opts=TokensTransformOpts())
            .tokens_to_text()
            .to_document_content_tuple()
            .tqdm()
            .to_dtm()
        )
        .single()
        .content
    )

    corpus.dump(tag="kallekulakurtkurt", folder=OUTPUT_FOLDER)
    assert isinstance(corpus, VectorizedCorpus)
    assert corpus.data.shape[0] == 5
    assert len(corpus.token2id) == corpus.data.shape[1]


# pylint: disable=too-many-locals
@pytest.mark.long_running
def test_compute_dtm_when_persist_is_false(config: CorpusConfig):

    args = ComputeOptsSpacyCSV(corpus_tag='compute_dtm_when_persist_is_false')

    corpus = workflows.document_term_matrix.compute(args=args, corpus_config=config)

    corpus.remove(tag=args.corpus_tag, folder=args.target_folder)
    corpus.dump(tag=args.corpus_tag, folder=args.target_folder)

    assert VectorizedCorpus.dump_exists(tag=args.corpus_tag, folder=args.target_folder)

    corpus_loaded = VectorizedCorpus.load(tag=args.corpus_tag, folder=args.target_folder)

    assert corpus_loaded is not None

    y_corpus = corpus.group_by_year()

    assert y_corpus is not None


def sample_corpus() -> VectorizedCorpus:
    corpus = [
        "the information we have is that the house had a tiny little mouse",
        "the sleeping cat saw the mouse",
        "the mouse ran away from the house",
        "the cat finally ate the mouse",
        "the end of the mouse story",
        "but a beginning of the cat story",
        "the fox saw the cat",
        "the fox ate the cat",
    ]

    document_index = pd.DataFrame(
        {
            'filename': [f'document_{i}.txt' for i in range(1, len(corpus) + 1)],
            'document_id': [i for i in range(0, len(corpus))],
            'document_name': [f'document_{i}' for i in range(1, len(corpus) + 1)],
        }
    )

    count_vectorizer = CountVectorizer()
    bag_term_matrix = count_vectorizer.fit_transform(corpus)

    corpus = VectorizedCorpus(
        bag_term_matrix,
        token2id=count_vectorizer.vocabulary_,
        document_index=document_index,
    )
    return corpus


def test_slice_py_regular_expressions():

    pattern = re.compile("^.*tion$")

    corpus: VectorizedCorpus = sample_corpus()

    sliced_corpus = corpus.slice_by(px=pattern.match)

    assert sliced_corpus is not None
    assert sliced_corpus.data.shape == (8, 1)
    assert len(sliced_corpus.token2id) == 1
    assert 'information' in sliced_corpus.token2id


def test_create_pipeline_by_string(config: CorpusConfig):
    cls_str = 'penelope.pipeline.spacy.pipelines.to_tagged_frame_pipeline'
    factory = create_pipeline_factory(cls_str)
    assert factory is not None
    p: CorpusPipeline = factory(corpus_config=config)
    assert p is not None
