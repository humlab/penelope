import os
from unittest.mock import Mock

import pandas as pd
import penelope.pipeline.spacy.convert as convert
import pytest
import spacy
from penelope.corpus import VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReader, TextReaderOpts, TextTransformOpts
from penelope.pipeline import CorpusConfig, CorpusPipeline, PipelinePayload, tagged_frame_to_tokens
from penelope.utility import PropertyValueMaskingOpts
from penelope.vendor.spacy import SPACY_DATA, prepend_path, prepend_spacy_path
from spacy.language import Language
from spacy.tokens import Doc

# pylint: disable=redefined-outer-name

TEST_CORPUS = [
    ('mars_1999_01.txt', 'Mars was once home to seas and oceans, and perhaps even life.'),
    ('mars_1999_02.txt', 'But its atmosphere has now been blown away.'),
    ('mars_1999_03.txt', 'Most activity beneath its surface has long ceased.'),
    ('mars_1999_04.txt', 'It’s a dead planet.'),
    ('mars_1999_05.txt', 'A volcano erupted on Mars 2.5 million years ago.'),
    ('mars_1999_06.txt', 'An eruption occurred as recently as 53,000 years ago in a region called Cerberus Fossae.'),
    ('mars_1999_07.txt', 'It is the youngest known volcanic eruption on Mars.'),
    ('mars_1999_08.txt', 'Some volcanos still erupts to the surface at rare intervals.'),
]
ATTRIBUTES = [
    "i",
    "text",
    "lemma",
    "lemma_",
    "pos_",
    "tag_",
    "dep_",
    "pos",
    "tag",
    "dep",
    "shape",
    "is_alpha",
    "is_stop",
    "is_punct",
    "is_space",
    "is_digit",
]


def spacy_model_path(model_name: str) -> str:
    path: str = os.path.join(os.environ.get("SPACY_DATA", ""), model_name)
    return path


@pytest.fixture(scope="module")
def en_nlp() -> Language:
    return spacy.load(os.path.join(os.environ.get("SPACY_DATA", ""), "en_core_web_sm"))


@pytest.fixture(scope="module")
def df_doc(en_nlp) -> Language:
    attributes = ["text", "lemma_", "pos_", "is_space", "is_punct", "is_digit", "is_alpha", "is_stop"]
    doc = convert.text_to_tagged_frame(
        TEST_CORPUS[0][1], attributes=attributes, attribute_value_filters=None, nlp=en_nlp
    )
    return doc


def test_annotate_document_with_lemma_and_pos_strings_succeeds(en_nlp):

    attributes = ["lemma_", "pos_"]

    df = convert.text_to_tagged_frame(
        TEST_CORPUS[0][1],
        attributes=attributes,
        attribute_value_filters=None,
        nlp=en_nlp,
    )

    assert df.columns.tolist() == attributes
    assert df.lemma_.tolist() == [
        'Mars',
        'be',
        'once',
        'home',
        'to',
        'sea',
        'and',
        'ocean',
        ',',
        'and',
        'perhaps',
        'even',
        'life',
        '.',
    ]
    assert df.pos_.tolist() == [
        'PROPN',
        'AUX',
        'ADV',
        'ADV',
        'ADP',
        'NOUN',
        'CCONJ',
        'NOUN',
        'PUNCT',
        'CCONJ',
        'ADV',
        'ADV',
        'NOUN',
        'PUNCT',
    ]


def test_annotate_document_with_lemma_and_pos_strings_and_attribute_value_filtersucceeds(en_nlp):

    attributes = ["lemma_", "pos_"]

    df = convert.text_to_tagged_frame(
        TEST_CORPUS[0][1],
        attributes=attributes,
        attribute_value_filters={'is_punct': False},
        nlp=en_nlp,
    )

    assert df.columns.tolist() == attributes
    assert df.lemma_.tolist() == [
        'Mars',
        'be',
        'once',
        'home',
        'to',
        'sea',
        'and',
        'ocean',
        'and',
        'perhaps',
        'even',
        'life',
    ]
    assert df.pos_.tolist() == [
        'PROPN',
        'AUX',
        'ADV',
        'ADV',
        'ADP',
        'NOUN',
        'CCONJ',
        'NOUN',
        'CCONJ',
        'ADV',
        'ADV',
        'NOUN',
    ]


@pytest.mark.skipif(SPACY_DATA == "", reason="SPACY_DATA not set")
def test_annotate_documents_with_lemma_and_pos_strings_succeeds():

    nlp = spacy.load(prepend_spacy_path("en_core_web_sm"))
    attributes = ["i", "text", "lemma_", "pos_"]

    dfs = convert.texts_to_tagged_frames(
        [text for _, text in TEST_CORPUS],
        attributes=attributes,
        attribute_value_filters=None,
        language=nlp,
    )
    df = next(dfs)

    assert df.columns.tolist() == attributes
    assert df.text.tolist() == [
        "Mars",
        "was",
        "once",
        "home",
        "to",
        "seas",
        "and",
        "oceans",
        ",",
        "and",
        "perhaps",
        "even",
        "life",
        ".",
    ]
    assert df.lemma_.tolist() == [
        'Mars',
        'be',
        'once',
        'home',
        'to',
        'sea',
        'and',
        'ocean',
        ',',
        'and',
        'perhaps',
        'even',
        'life',
        '.',
    ]
    assert df.pos_.tolist() == [
        'PROPN',
        'AUX',
        'ADV',
        'ADV',
        'ADP',
        'NOUN',
        'CCONJ',
        'NOUN',
        'PUNCT',
        'CCONJ',
        'ADV',
        'ADV',
        'NOUN',
        'PUNCT',
    ]


def test_extract_tokens_when_punct_filter_is_disabled_succeeds(df_doc):

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True)
    filter_opts = PropertyValueMaskingOpts(is_punct=None)
    tokens = tagged_frame_to_tokens(doc=df_doc, extract_opts=extract_opts, filter_opts=filter_opts)
    assert tokens == [
        'mars',
        'be',
        'once',
        'home',
        'to',
        'sea',
        'and',
        'ocean',
        ',',
        'and',
        'perhaps',
        'even',
        'life',
        '.',
    ]


def test_extract_tokens_when_lemma_lacks_underscore_succeeds(df_doc):

    extract_opts = ExtractTaggedTokensOpts(lemmatize=False, target_override="lemma_")
    filter_opts = PropertyValueMaskingOpts(is_punct=False)
    tokens = tagged_frame_to_tokens(doc=df_doc, extract_opts=extract_opts, filter_opts=filter_opts)
    assert tokens == ['Mars', 'be', 'once', 'home', 'to', 'sea', 'and', 'ocean', 'and', 'perhaps', 'even', 'life']


def test_extract_tokens_target_text_succeeds(df_doc):
    extract_opts = ExtractTaggedTokensOpts(lemmatize=False)
    filter_opts = PropertyValueMaskingOpts(is_punct=False)
    tokens = tagged_frame_to_tokens(doc=df_doc, extract_opts=extract_opts, filter_opts=filter_opts)
    assert tokens == ["Mars", "was", "once", "home", "to", "seas", "and", "oceans", "and", "perhaps", "even", "life"]


def test_extract_tokens_lemma_no_stops_succeeds(df_doc):
    extract_opts = ExtractTaggedTokensOpts(lemmatize=True)
    filter_opts = PropertyValueMaskingOpts(is_stop=False, is_punct=False)
    tokens = tagged_frame_to_tokens(doc=df_doc, extract_opts=extract_opts, filter_opts=filter_opts)
    assert tokens == ['mars', 'home', 'sea', 'ocean', 'life']


def test_extract_tokens_pos_propn_succeeds(df_doc):
    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes='|PROPN|', pos_paddings=None)
    filter_opts = PropertyValueMaskingOpts(is_punct=False)
    tokens = tagged_frame_to_tokens(doc=df_doc, extract_opts=extract_opts, filter_opts=filter_opts)
    assert tokens == ['mars']


def test_extract_tokens_pos_verb_noun_text_succeeds(df_doc):
    extract_opts = ExtractTaggedTokensOpts(lemmatize=False, pos_includes='|VERB|NOUN|', pos_paddings=None)
    filter_opts = PropertyValueMaskingOpts(is_punct=False)
    tokens = tagged_frame_to_tokens(doc=df_doc, extract_opts=extract_opts, filter_opts=filter_opts)
    assert tokens == ['seas', 'oceans', 'life']


def dummy_source():
    test_corpus = [
        ('tran_2019_01_test.txt', 'a b c c'),
        ('tran_2019_02_test.txt', 'a a b d'),
        ('tran_2019_03_test.txt', 'a e e b'),
        ('tran_2020_01_test.txt', 'c c d a'),
        ('tran_2020_02_test.txt', 'a b b e'),
    ]
    return test_corpus


def test_spacy_pipeline_load_text_resolves():
    reader_opts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    source = dummy_source()
    config = Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload(source=source))
    pipeline = CorpusPipeline(config=config).load_text(reader_opts=reader_opts)

    payloads = [x.content for x in pipeline.resolve()]

    assert payloads == [x[1] for x in source]
    assert len(pipeline.payload.document_index) == len(source)
    assert all(pipeline.payload.document_index.filename == [x[0] for x in source])


def test_spacy_pipeline_load_text_to_spacy_doc_resolves(en_nlp):
    reader_opts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    source = dummy_source()
    config = Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload(source=source).put2(pos_column="pos_"))
    pipeline = CorpusPipeline(config=config).set_spacy_model(en_nlp).load_text(reader_opts=reader_opts).text_to_spacy()

    payloads = [x.content for x in pipeline.resolve()]

    assert all(isinstance(x, Doc) for x in payloads)


def test_spacy_pipeline_load_text_to_spacy_to_dataframe_resolves(en_nlp):
    reader_opts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    reader = TextReader.create(TEST_CORPUS, reader_opts=reader_opts)
    config = Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload(source=reader).put2(pos_column="pos_"))
    attributes = ['text', 'lemma_', 'pos_']
    pipeline = (
        CorpusPipeline(config=config)
        .set_spacy_model(en_nlp)
        .load_text(reader_opts=reader_opts)
        .text_to_spacy()
        .spacy_to_tagged_frame(attributes=attributes)
    )

    payloads = [x.content for x in pipeline.resolve()]

    assert all(isinstance(x, pd.DataFrame) for x in payloads)
    assert all(x.columns.tolist() == attributes for x in payloads)


def test_spacy_pipeline_load_text_to_spacy_to_dataframe_to_tokens_resolves(en_nlp):

    reader_opts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    text_transform_opts = TextTransformOpts()
    reader = TextReader.create(TEST_CORPUS, reader_opts=reader_opts, transform_opts=text_transform_opts)

    config = Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload(source=reader).put2(pos_column="pos_"))
    attributes = ['text', 'lemma_', 'pos_']
    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|VERB|NOUN|',
        pos_paddings='|ADJ|',
    )
    transform_opts = None
    filter_opts = PropertyValueMaskingOpts(is_punct=False)
    pipeline = (
        CorpusPipeline(config=config)
        .load_text(reader_opts=reader_opts)
        .set_spacy_model(en_nlp)
        .text_to_spacy()
        .spacy_to_tagged_frame(attributes=attributes)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
    )

    payloads = [x.content for x in pipeline.resolve()]

    assert payloads == [
        ['sea', 'ocean', 'life'],
        ['atmosphere', 'blow'],
        ['*', 'activity', 'surface', 'cease'],
        ['*', 'planet'],
        ['volcano', 'erupt', 'year'],
        ['eruption', 'occur', 'year', 'region', 'call'],
        ['*', '*', 'eruption'],
        ['volcanos', 'erupt', 'surface', '*', 'interval'],
    ]

    assert set(list(pipeline.payload.document_index.columns)) == set(
        [
            'filename',
            'year',
            'document_id',
            'document_name',
            'Adverb',
            'Conjunction',
            'Delimiter',
            'Noun',
            'Other',
            'Preposition',
            'n_tokens',
            'n_raw_tokens',
            'Pronoun',
            'Verb',
            'Adjective',
            'Numeral',
        ]
    )


def test_spacy_pipeline_load_text_to_spacy_to_dataframe_to_tokens_to_text_to_dtm(en_nlp):

    reader_opts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    text_transform_opts = TextTransformOpts()
    reader = TextReader.create(TEST_CORPUS, reader_opts=reader_opts, transform_opts=text_transform_opts)

    attributes = ['text', 'lemma_', 'pos_', 'is_punct']
    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|VERB|NOUN|',
        pos_paddings=None,
    )
    transform_opts = None
    filter_opts = PropertyValueMaskingOpts(is_punct=False)
    vectorize_opts = VectorizeOpts(verbose=True)

    config = Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload(source=reader).put2(pos_column='pos_'))

    pipeline = (
        CorpusPipeline(config=config)
        .load_text(reader_opts=reader_opts, transform_opts=text_transform_opts)
        .set_spacy_model(en_nlp)
        .text_to_spacy()
        .spacy_to_tagged_frame(attributes=attributes)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
        .tokens_to_text()
        .to_document_content_tuple()
        .to_dtm(vectorize_opts)
    )

    corpus = pipeline.value()
    assert corpus is not None
    assert isinstance(corpus, VectorizedCorpus)


def test_spacy_pipeline_extract_text_to_vectorized_corpus(en_nlp):

    reader_opts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    text_transform_opts = TextTransformOpts()
    reader = TextReader.create(TEST_CORPUS, reader_opts=reader_opts, transform_opts=text_transform_opts)

    attributes = ['text', 'lemma_', 'pos_', 'is_punct']
    tagged_columns = {'text_column': 'text', 'lemma_column': 'lemma_', 'pos_column': 'pos_'}
    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|VERB|NOUN|',
        pos_paddings=None,
    )
    transform_opts = None
    filter_opts = PropertyValueMaskingOpts(is_punct=False)
    vectorize_opts = VectorizeOpts(verbose=True)

    config = Mock(spec=CorpusConfig, pipeline_payload=PipelinePayload(source=reader).put2(**tagged_columns))

    pipeline = (
        CorpusPipeline(config=config)
        .load_text(reader_opts=reader_opts, transform_opts=text_transform_opts)
        .set_spacy_model(en_nlp)
        .text_to_spacy()
        .spacy_to_tagged_frame(attributes=attributes)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=transform_opts)
        .tokens_to_text()
        .to_document_content_tuple()
        .to_dtm(vectorize_opts)
    )

    corpus = pipeline.value()

    assert isinstance(corpus, VectorizedCorpus)


@pytest.mark.skipif(SPACY_DATA == "", reason="SPACY_DATA not set")
def test_spacy_data_location():

    spacy_data: str = "/data/lib/spacy_data/2.3.1"

    if not os.path.isdir(spacy_data):
        return

    nlp: Language = spacy.load(os.path.join(spacy_data, "en_core_web_sm"))
    assert isinstance(nlp, Language)

    nlp: Language = spacy.load(os.path.join(spacy_data, "en"))
    assert isinstance(nlp, Language)

    nlp: Language = spacy.load(prepend_path("en_core_web_sm", spacy_data))
    assert isinstance(nlp, Language)

    nlp: Language = spacy.load(prepend_path(nlp, ""))
    assert isinstance(nlp, Language)
