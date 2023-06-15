from os.path import join

import pandas as pd

from penelope import corpus as corpora
from penelope import pipeline, utility
from penelope.common.render_text import LemmaCorpusLoader, TextCorpusLoader, TokenizedCorpusLoader, ZipLoader
from penelope.corpus import load_document_index
from penelope.corpus.transforms import dehyphen
from penelope.pipeline import tasks

# FIXME: Create a test fixture for this
TEST_CORPUS_FILENAME: str = '/data/inidun/courier/corpus/v0.2.0/article_corpus.zip'


def test_remove_hyphens():
    text: str = """The choreo-
 graphy which makes one's
head spin in its com¬
plex pat-

terns has
been evolved through hun-dreds of years
of tra-
    dition and training."""

    result = dehyphen(text)

    assert (
        result
        == """The choreography
which makes one's
head spin in its complex
patterns
has
been evolved through hun-dreds of years
of tradition
and training."""
    )


def test_load():
    text_transform_opts: corpora.TextTransformOpts = corpora.TextTransformOpts(
        transforms="normalize-whitespaces,dehyphen,strip-accents"
    )
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.create(
        corpus_name='courier',
        corpus_type=pipeline.CorpusType.Text,
        corpus_pattern='*.zip',
        checkpoint_opts=pipeline.CheckpointOpts(),
        text_reader_opts=corpora.TextReaderOpts(),
        pipelines={},
        pipeline_payload=pipeline.PipelinePayload(),
        language='english',
        text_transform_opts=text_transform_opts,
    )

    text: str = utility.read_textfile('./tests/test_data/courier/1955_069029_026.txt')

    # config.pipeline_payload.source = [('apa.txt', text)]

    task = tasks.LoadText(
        source=[('apa.txt', text)],
        reader_opts=config.text_reader_opts,
        transform_opts=text_transform_opts,
    )

    p = pipeline.CorpusPipeline(config=config).add(task).setup()

    processed_payload: pipeline.DocumentPayload = p.single()

    # payload: pipeline.DocumentPayload = pipeline.DocumentPayload(
    #     content_type=pipeline.ContentType.TEXT, filename='APA.txt', content=text
    # )
    # processed_payload: pipeline.DocumentPayload = task.process_payload(payload)

    assert processed_payload is not None


TEST_DOCUMENT_INFO = {
    'document_id': 1774,
    'courier_id': 78179,
    'year': 1961,
    'record_number': 63805,
    'pages': '[32]',
    'catalogue_title': "They call their telescope 'Bima Sakti'",
    'authors': 'Blanco, Victor M.',
    'filename': '1961_078179_63805.txt',
    'document_name': '1961_078179_63805',
}


def test_document_index():
    source: str = TEST_CORPUS_FILENAME

    di: pd.DataFrame = load_document_index(join(source, 'document_index.csv'), sep="\t")

    assert di is not None

    assert set(di.columns) == set(TEST_DOCUMENT_INFO.keys())

    assert di.loc["1961_078179_63805"].todict() == TEST_DOCUMENT_INFO


def test_load_article_document_text():
    info: dict = dict(TEST_DOCUMENT_INFO)

    # article_name: str = "1961_078179_63805"
    article_name: str = f'{info.get("year")}_{int(info.get("courier_id",0)):06}_{info.get("record_number")}'

    source: str = TEST_CORPUS_FILENAME

    text = ZipLoader(source).load(article_name)

    assert text is not None


def test_load_article_document_using_loader():
    source: str = TEST_CORPUS_FILENAME

    di: pd.DataFrame = ZipLoader(source).load_document_index()

    assert di is not None


def test_load_tagged_article_document_text():
    info: dict = dict(TEST_DOCUMENT_INFO)

    # article_name: str = "1961_078179_63805"
    article_name: str = f'{info.get("year")}_{int(info.get("courier_id",0)):06}_{info.get("record_number")}'

    source: str = TEST_CORPUS_FILENAME

    text = ZipLoader(source).load(f'{article_name}.txt')
    assert (text or '').startswith('They call their telescope\nJava')

    text = TextCorpusLoader(source).load(article_name)
    assert (text or '').startswith('They call their telescope\nJava')

    text = TokenizedCorpusLoader(source).load(article_name)
    assert (text or '').startswith("They call their telescope Java 's")

    text = LemmaCorpusLoader(source).load(article_name)
    assert (text or '').startswith("they call their telescope Java 's")
