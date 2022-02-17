import contextlib
import os
import pathlib
import re
import uuid
from typing import List

import pandas as pd
import pytest

import penelope.workflows.vectorize.dtm as workflow
from penelope import corpus as corpora
from penelope import pipeline, utility
from penelope.pipeline import tasks
from penelope.pipeline.spacy import pipelines as spacy_pipeline
from penelope.vendor import spacy_api
from penelope.workflows.interface import ComputeOpts
from tests.utils import OUTPUT_FOLDER, inline_code

try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    ...


CORPUS_FOLDER = './tests/test_data'

# pylint: disable=redefined-outer-name


def fake_config() -> pipeline.CorpusConfig:

    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load('./tests/test_data/SSI.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    return corpus_config


@pytest.fixture(scope='module')
def config():
    return fake_config()


def test_corpus_config_set_folder(config: pipeline.CorpusConfig):

    current_source = config.pipeline_payload.source
    config.pipeline_payload.folders(CORPUS_FOLDER)

    assert config.pipeline_payload.source == current_source


def test_load_text_returns_payload_with_expected_document_index(config: pipeline.CorpusConfig):

    transform_opts = corpora.TextTransformOpts()

    pipe = pipeline.CorpusPipeline(config=config).load_text(
        reader_opts=config.text_reader_opts, transform_opts=transform_opts
    )
    assert pipe is not None

    payloads: List[pipeline.DocumentPayload] = [x for x in pipe.resolve()]

    assert len(payloads) == 5
    assert len(pipe.payload.document_index) == 5
    assert len(pipe.payload.metadata) == 5
    assert pipe.payload.pos_schema_name == "Universal"
    assert pipe.payload.get('text_reader_opts') == config.text_reader_opts.props
    assert isinstance(pipe.payload.document_index, pd.DataFrame)

    columns = pipe.payload.document_index.columns.tolist()
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
    assert pipe.payload.document_lookup('RECOMMENDATION_0201_049455_2017.txt')['unesco_id'] == 49455


@pytest.mark.skipif(not spacy_api.SPACY_INSTALLED, reason="spaCy not installed")
@pytest.mark.long_running
def test_pipeline_load_text_tag_checkpoint_stores_checkpoint(config: pipeline.CorpusConfig):

    tagged_corpus_source: str = os.path.join(OUTPUT_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    transform_opts = corpora.TextTransformOpts()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pathlib.Path(tagged_corpus_source).unlink(missing_ok=True)

    _ = (
        pipeline.CorpusPipeline(config=config)
        .set_spacy_model(config.pipeline_payload.memory_store['spacy_model'])
        .load_text(reader_opts=config.text_reader_opts, transform_opts=transform_opts)
        .text_to_spacy()
        .tqdm()
        .spacy_to_pos_tagged_frame()
        .checkpoint(tagged_corpus_source, force_checkpoint=False)
    ).exhaust()

    assert os.path.isfile(tagged_corpus_source)
    pathlib.Path(tagged_corpus_source).unlink(missing_ok=True)


def test_pipeline_can_load_pos_tagged_checkpoint(config: pipeline.CorpusConfig):

    tagged_corpus_source: str = os.path.join(CORPUS_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    pipe = pipeline.CorpusPipeline(config=config).checkpoint(tagged_corpus_source, force_checkpoint=False)

    payloads: List[pipeline.DocumentPayload] = pipe.to_list()

    assert len(payloads) == 5
    assert len(pipe.payload.document_index) == 5
    assert isinstance(pipe.payload.document_index, pd.DataFrame)


@pytest.mark.long_running
def test_pipeline_tagged_frame_to_tokens_succeeds(config: pipeline.CorpusConfig):

    tagged_corpus_source: str = os.path.join(CORPUS_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|NOUN|',
        pos_paddings=None,
        **config.pipeline_payload.tagged_columns_names,
        filter_opts=dict(is_punct=False),
    )

    tagged_payload = next(
        pipeline.CorpusPipeline(config=config).checkpoint(tagged_corpus_source, force_checkpoint=False).resolve()
    )

    tokens_payload = next(
        pipeline.CorpusPipeline(config=config)
        .checkpoint(tagged_corpus_source, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=None)
        .resolve()
    )

    assert tagged_payload.filename == tokens_payload.filename
    assert len(tagged_payload.content[tagged_payload.content.pos_ == 'NOUN']) == len(tokens_payload.content)


@pytest.mark.long_running
def test_pipeline_tagged_frame_to_vocabulary_succeeds(config: pipeline.CorpusConfig):

    tagged_corpus_source: str = os.path.join(CORPUS_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    pipe: pipeline.CorpusPipeline = (
        pipeline.CorpusPipeline(config=config)
        .checkpoint(tagged_corpus_source, force_checkpoint=False)
        .vocabulary(lemmatize=True, progress=False)
        .exhaust()
    )

    assert pipe.payload.token2id is not None
    assert pipe.payload.token2id.tf is not None
    assert len(pipe.payload.token2id) == 1147
    assert len(pipe.payload.token2id) == len(pipe.payload.token2id.tf) is not None
    assert set(pipe.payload.token2id.data.keys()) == {x.lower() for x in pipe.payload.token2id.keys()}
    assert 'Cultural' not in pipe.payload.token2id
    assert 'wars' not in pipe.payload.token2id
    assert 'war' in pipe.payload.token2id

    pipe: pipeline.CorpusPipeline = (
        pipeline.CorpusPipeline(config=config)
        .checkpoint(tagged_corpus_source, force_checkpoint=False)
        .vocabulary(lemmatize=False, progress=False)
        .exhaust()
    )

    assert len(pipe.payload.token2id) == 1478
    assert 'Cultural' in pipe.payload.token2id
    assert 'wars' in pipe.payload.token2id

    assert pipe.payload.token2id.tf[pipe.payload.token2id['the']] == 704


@pytest.mark.long_running
def test_pipeline_tagged_frame_to_text_succeeds(config: pipeline.CorpusConfig):

    tagged_corpus_source: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|NOUN|',
        pos_paddings=None,
        **config.pipeline_payload.tagged_columns_names,
        filter_opts=dict(is_punct=False),
    )

    tagged_payload = next(
        pipeline.CorpusPipeline(config=config).checkpoint(tagged_corpus_source, force_checkpoint=False).resolve()
    )

    text_payload = next(
        pipeline.CorpusPipeline(config=config)
        .checkpoint(tagged_corpus_source, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=None)
        .tokens_to_text()
        .resolve()
    )

    assert tagged_payload.filename == text_payload.filename
    assert len(tagged_payload.content[tagged_payload.content.pos_ == 'NOUN']) == len(text_payload.content.split())


def test_pipeline_take_succeeds(config: pipeline.CorpusConfig):
    tagged_corpus_source: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
        lemmatize=True, **config.pipeline_payload.tagged_columns_names, filter_opts=dict(is_punct=False)
    )

    take_payloads = (
        pipeline.CorpusPipeline(config=config)
        .checkpoint(tagged_corpus_source, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=None)
        .tokens_to_text()
        .take(2)
    )

    assert len(take_payloads) == 2


def test_pipeline_tagged_frame_to_tuple_succeeds(config: pipeline.CorpusConfig):

    tagged_corpus_source: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|NOUN|',
        pos_paddings='|VERB|',
        **config.pipeline_payload.tagged_columns_names,
        filter_opts=dict(is_punct=False),
    )

    payloads = (
        pipeline.CorpusPipeline(config=config)
        .checkpoint(tagged_corpus_source, force_checkpoint=False)
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=None)
        .tokens_to_text()
        .to_list()
    )
    assert len(payloads) == 5

    assert all(isinstance(payload.content, str) for payload in payloads)


def test_pipeline_find_task(config: pipeline.CorpusConfig):
    p: pipeline.CorpusPipeline = (
        pipeline.CorpusPipeline(config=config).checkpoint("dummy_name", force_checkpoint=False).tqdm()
    )
    assert isinstance(p.find(tasks.Checkpoint), tasks.Checkpoint)
    assert isinstance(p.find(tasks.Tqdm), tasks.Tqdm)


def test_pipeline_text_to_dtm_succeeds(config: pipeline.CorpusConfig):

    target_tag: str = uuid.uuid1()

    tagged_corpus_source: str = os.path.join(CORPUS_FOLDER, 'checkpoint_pos_tagged_test.zip')

    extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|NOUN|',
        pos_paddings=None,
        **config.pipeline_payload.tagged_columns_names,
        filter_opts=dict(is_punct=False),
    )

    corpus: corpora.VectorizedCorpus = (
        (
            pipeline.CorpusPipeline(config=config)
            .checkpoint(tagged_corpus_source, force_checkpoint=False)
            .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=None)
            .tokens_transform(transform_opts=corpora.TokensTransformOpts())
            .tokens_to_text()
            .tqdm()
            .to_dtm()
        )
        .single()
        .content
    )

    corpus.dump(tag=target_tag, folder=OUTPUT_FOLDER)

    assert isinstance(corpus, corpora.VectorizedCorpus)
    assert corpus.data.shape[0] == 5
    assert len(corpus.token2id) == corpus.data.shape[1]

    corpus.remove(tag=target_tag, folder=OUTPUT_FOLDER)


# pylint: disable=too-many-locals
@pytest.mark.skipif(not spacy_api.SPACY_INSTALLED, reason="spaCy not installed")
@pytest.mark.long_running
def test_workflow_to_dtm_step_by_step(config: pipeline.CorpusConfig):

    corpus_tag: str = uuid.uuid1()
    target_folder: str = "./tests/output"
    corpus_source: str = './tests/test_data/legal_instrument_five_docs_test.zip'
    tagged_corpus_source: str = f"./tests/output/{uuid.uuid1()}_pos_csv.zip"

    args: ComputeOpts = ComputeOpts(
        corpus_tag=corpus_tag,
        corpus_source=corpus_source,
        target_folder=target_folder,
        corpus_type=pipeline.CorpusType.SpacyCSV,
        # pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
        transform_opts=corpora.TokensTransformOpts(language='english', remove_stopwords=True, to_lower=True),
        text_reader_opts=corpora.TextReaderOpts(filename_pattern='*.csv', filename_fields=['year:_:1']),
        extract_opts=corpora.ExtractTaggedTokensOpts(
            lemmatize=True,
            pos_includes='|NOUN|PROPN|VERB|',
            pos_excludes='|PUNCT|EOL|SPACE|',
            **config.pipeline_payload.tagged_columns_names,
            filter_opts=dict(is_alpha=False, is_punct=False, is_space=False),
        ),
        create_subfolder=False,
        persist=True,
        tf_threshold=1,
        tf_threshold_mask=False,
        vectorize_opts=corpora.VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            min_tf=1,
            max_tokens=None,
        ),
        enable_checkpoint=True,
        force_checkpoint=True,
    )
    with inline_code(spacy_pipeline.to_tagged_frame_pipeline):

        tagged_frame_filename: str = tagged_corpus_source or utility.path_add_suffix(
            config.pipeline_payload.source, '_pos_csv'
        )

        p: pipeline.CorpusPipeline = (
            pipeline.CorpusPipeline(config=config)
            .set_spacy_model(config.pipeline_payload.memory_store['spacy_model'])
            .load_text(
                reader_opts=config.text_reader_opts,
                transform_opts=None,
                source=corpus_source,
            )
            .text_to_spacy()
            .spacy_to_pos_tagged_frame()
            .checkpoint(filename=tagged_frame_filename, force_checkpoint=args.force_checkpoint)
        )

        if args.enable_checkpoint:
            p = p.checkpoint_feather(folder=config.get_feather_folder(corpus_source), force=args.force_checkpoint)

        p.exhaust()


@pytest.mark.skipif(not spacy_api.SPACY_INSTALLED, reason="spaCy not installed")
@pytest.mark.long_running
def test_workflow_to_dtm(config: pipeline.CorpusConfig):

    args: ComputeOpts = ComputeOpts(
        corpus_tag=f'{uuid.uuid1()}',
        corpus_source='./tests/test_data/legal_instrument_five_docs_test.zip',
        corpus_type=pipeline.CorpusType.Text,
        target_folder='./tests/output/',
        transform_opts=corpora.TokensTransformOpts(language='english', remove_stopwords=True, to_lower=True),
        text_reader_opts=corpora.TextReaderOpts(filename_pattern='*.csv', filename_fields=['year:_:1']),
        extract_opts=corpora.ExtractTaggedTokensOpts(
            lemmatize=True,
            pos_includes='|NOUN|PROPN|VERB|',
            pos_excludes='|PUNCT|EOL|SPACE|',
            **config.pipeline_payload.tagged_columns_names,
            filter_opts=dict(is_alpha=False, is_punct=False, is_space=False),
        ),
        vectorize_opts=corpora.VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            min_tf=1,
            max_tokens=None,
        ),
        create_subfolder=False,
        persist=True,
        enable_checkpoint=True,
        force_checkpoint=True,
        tf_threshold=1,
        tf_threshold_mask=False,
        tagged_corpus_source='./tests/output/legal_instrument_five_docs_test_pos_csv.zip',
    )

    corpus = workflow.compute(args=args, corpus_config=config)

    corpus.remove(tag=args.corpus_tag, folder=args.target_folder)
    corpus.dump(tag=args.corpus_tag, folder=args.target_folder)

    assert corpora.VectorizedCorpus.dump_exists(tag=args.corpus_tag, folder=args.target_folder)

    corpus_loaded = corpora.VectorizedCorpus.load(tag=args.corpus_tag, folder=args.target_folder)

    assert corpus_loaded is not None

    y_corpus = corpus.group_by_year()

    assert y_corpus is not None

    with contextlib.suppress(Exception):
        corpus.remove(tag=args.corpus_tag, folder=args.target_folder)


def sample_corpus() -> corpora.VectorizedCorpus:
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

    corpus = corpora.VectorizedCorpus(
        bag_term_matrix,
        token2id=count_vectorizer.vocabulary_,
        document_index=document_index,
    )
    return corpus


def test_slice_py_regular_expressions():

    pattern = re.compile("^.*tion$")

    corpus: corpora.VectorizedCorpus = sample_corpus()

    sliced_corpus = corpus.slice_by(px=pattern.match)

    assert sliced_corpus is not None
    assert sliced_corpus.data.shape == (8, 1)
    assert len(sliced_corpus.token2id) == 1
    assert 'information' in sliced_corpus.token2id


def test_create_pipeline_by_string(config: pipeline.CorpusConfig):
    cls_str = 'penelope.pipeline.spacy.pipelines.to_tagged_frame_pipeline'
    factory = pipeline.create_pipeline_factory(cls_str)
    assert factory is not None
    p: pipeline.CorpusPipeline = factory(corpus_config=config)
    assert p is not None
