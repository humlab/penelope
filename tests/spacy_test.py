import pandas as pd
import pytest
import spacy
from penelope.corpus import VectorizeOpts
from penelope.corpus.readers import TextReader, streamify_text_source
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.vendor.spacy.extract import (
    ExtractTextOpts,
    dataframe_to_tokens,
    text_to_annotated_dataframe,
    texts_to_annotated_dataframes,
)
from penelope.vendor.spacy.pipeline import PipelinePayload, SpacyPipeline
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


def test_annotate_document_with_lemma_and_pos_strings_succeeds():

    nlp = spacy.load("en_core_web_sm")
    attributes = ["lemma_", "pos_"]

    df = text_to_annotated_dataframe(
        TEST_CORPUS[0][1],
        attributes=attributes,
        nlp=nlp,
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


def test_annotate_documents_with_lemma_and_pos_strings_succeeds():

    nlp = spacy.load("en_core_web_sm")
    attributes = ["i", "text", "lemma_", "pos_"]

    dfs = texts_to_annotated_dataframes(
        [text for _, text in TEST_CORPUS],
        attributes=attributes,
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


@pytest.fixture(scope="module")
def en_nlp() -> Language:
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def df_doc() -> Language:
    nlp = spacy.load("en_core_web_sm")
    attributes = ["text", "lemma_", "pos_", "is_space", "is_punct", "is_digit", "is_alpha", "is_stop"]
    doc = text_to_annotated_dataframe(
        TEST_CORPUS[0][1],
        attributes=attributes,
        nlp=nlp,
    )
    return doc


def test_extract_tokens_when_punct_filter_enables_succeeds(df_doc):

    extract_opts = ExtractTextOpts(target="lemma", is_punct=True, is_space=False)
    tokens = dataframe_to_tokens(doc=df_doc, extract_opts=extract_opts)
    assert tokens == [
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


def test_extract_tokens_when_lemma_lacks_underscore_succeeds(df_doc):

    extract_opts = ExtractTextOpts(target="lemma", is_punct=False, is_space=False)
    tokens = dataframe_to_tokens(doc=df_doc, extract_opts=extract_opts)
    assert tokens == ['Mars', 'be', 'once', 'home', 'to', 'sea', 'and', 'ocean', 'and', 'perhaps', 'even', 'life']


def test_extract_tokens_target_text_succeeds(df_doc):
    extract_opts = ExtractTextOpts(target="text", is_punct=False, is_space=False)
    tokens = dataframe_to_tokens(doc=df_doc, extract_opts=extract_opts)
    assert tokens == ["Mars", "was", "once", "home", "to", "seas", "and", "oceans", "and", "perhaps", "even", "life"]


def test_extract_tokens_lemma_no_stops_succeeds(df_doc):
    extract_opts = ExtractTextOpts(target="lemma", is_stop=False, is_punct=False, is_space=False)
    tokens = dataframe_to_tokens(doc=df_doc, extract_opts=extract_opts)
    assert tokens == ['Mars', 'home', 'sea', 'ocean', 'life']


def test_extract_tokens_pos_propn_succeeds(df_doc):
    extract_opts = ExtractTextOpts(target="lemma", include_pos={'PROPN'})
    tokens = dataframe_to_tokens(doc=df_doc, extract_opts=extract_opts)
    assert tokens == ['Mars']


def test_extract_tokens_pos_verb_noun_text_succeeds(df_doc):
    extract_opts = ExtractTextOpts(target="text", include_pos={'VERB', 'NOUN'})
    tokens = dataframe_to_tokens(doc=df_doc, extract_opts=extract_opts)
    assert tokens == ['seas', 'oceans', 'life']


def dummy_source():
    test_corpus = [
        ('tran_2019_01_test.txt', 'a b c c'),
        ('tran_2019_02_test.txt', 'a a b d'),
        ('tran_2019_03_test.txt', 'a e e b'),
        ('tran_2020_01_test.txt', 'c c d a'),
        ('tran_2020_02_test.txt', 'a b b e'),
    ]
    source = streamify_text_source(test_corpus)
    return source


def test_spacy_pipeline_load_text_resolves():
    source = dummy_source()
    payload = PipelinePayload(source=source, document_index=None)
    pipeline = SpacyPipeline(payload=payload).load(filename_pattern="*.txt", filename_fields="year:_:1")

    payloads = [x.content for x in pipeline.resolve()]

    assert payloads == [x[1] for x in source]
    assert len(pipeline.payload.document_index) == len(source)
    assert all(pipeline.payload.document_index.filename == [x[0] for x in source])


def test_spacy_pipeline_load_text_to_spacy_doc_resolves(en_nlp):
    source = dummy_source()
    payload = PipelinePayload(source=source, document_index=None)
    pipeline = (
        SpacyPipeline(payload=payload)
        .load(filename_pattern="*.txt", filename_fields="year:_:1")
        .text_to_spacy(nlp=en_nlp)
    )

    payloads = [x.content for x in pipeline.resolve()]

    assert all([isinstance(x, Doc) for x in payloads])


def test_spacy_pipeline_load_text_to_spacy_to_dataframe_resolves(en_nlp):
    reader = TextReader(TEST_CORPUS, filename_pattern="*.txt", filename_fields="year:_:1")
    payload = PipelinePayload(source=reader, document_index=None)
    attributes = ['text', 'lemma_', 'pos_']
    pipeline = (
        SpacyPipeline(payload=payload)
        .load(filename_pattern="*.txt", filename_fields="year:_:1")
        .text_to_spacy(nlp=en_nlp)
        .spacy_to_dataframe(en_nlp, attributes=attributes)
    )

    payloads = [x.content for x in pipeline.resolve()]

    assert all([isinstance(x, pd.DataFrame) for x in payloads])
    assert all([len(x) > 0 for x in payloads])
    assert all([x.columns.tolist() == attributes for x in payloads])


def test_spacy_pipeline_load_text_to_spacy_to_dataframe_to_tokensresolves(en_nlp):
    reader = TextReader(TEST_CORPUS, filename_pattern="*.txt", filename_fields="year:_:1")
    payload = PipelinePayload(source=reader, document_index=None)
    attributes = ['text', 'lemma_', 'pos_']
    extract_text_opts = ExtractTextOpts(
        target="lemma",
        include_pos={'VERB', 'NOUN'},
    )
    pipeline = (
        SpacyPipeline(payload=payload)
        .load(filename_pattern="*.txt", filename_fields="year:_:1")
        .text_to_spacy(nlp=en_nlp)
        .spacy_to_dataframe(en_nlp, attributes=attributes)
        .dataframe_to_tokens(extract_text_opts=extract_text_opts)
    )

    payloads = [x.content for x in pipeline.resolve()]

    assert payloads == [
        ['sea', 'ocean', 'life'],
        ['atmosphere', 'blow'],
        ['activity', 'surface', 'cease'],
        ['planet'],
        ['volcano', 'erupt', 'year'],
        ['eruption', 'occur', 'year', 'region', 'call'],
        ['eruption'],
        ['volcanos', 'erupt', 'surface', 'interval'],
    ]


def test_spacy_pipeline_load_text_to_spacy_to_dataframe_to_tokens_to_text_to_dtm(en_nlp):
    reader = TextReader(TEST_CORPUS, filename_pattern="*.txt", filename_fields="year:_:1")
    payload = PipelinePayload(source=reader, document_index=None)
    attributes = ['text', 'lemma_', 'pos_']
    extract_text_opts = ExtractTextOpts(
        target="lemma",
        include_pos={'VERB', 'NOUN'},
    )
    vectorize_opts = VectorizeOpts(verbose=True)
    pipeline = (
        SpacyPipeline(payload=payload)
        .load(filename_pattern="*.txt", filename_fields="year:_:1")
        .text_to_spacy(nlp=en_nlp)
        .spacy_to_dataframe(en_nlp, attributes=attributes)
        .dataframe_to_tokens(extract_text_opts=extract_text_opts)
        .tokens_to_text()
        .to_dtm(vectorize_opts)
    )

    corpus = pipeline.resolve()

    assert isinstance(corpus, VectorizedCorpus)
