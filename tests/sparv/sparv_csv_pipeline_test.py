import pytest

from penelope import pipeline
from penelope.corpus import ExtractTaggedTokensOpts
from penelope.corpus.readers import tng
from penelope.pipeline import tagged_frame, tasks

# pylint: disable=redefined-outer-name


@pytest.fixture
def corpus_config():
    config = pipeline.CorpusConfig.load('./tests/test_data/riksdagens_protokoll/riksdagens-protokoll.yml')

    config.pipeline_payload.source = './tests/test_data/riksdagens_protokoll/riksdagens-protokoll.test.sparv4.csv.zip'
    config.pipeline_payload.document_index_source = None

    return config


def test_read_sparv_csv_zip_using_tng_zip_source(corpus_config: pipeline.CorpusConfig):
    with tng.ZipSource(source_path=corpus_config.pipeline_payload.source) as source:
        names = source.namelist()
        texts = [source.read(filename) for filename in source.namelist()]

    assert names == ['prot_197677__25.csv', 'prot_197677__26.csv', 'prot_197677__27.csv']
    assert len(texts) == 3


def test_read_sparv_csv_zip_using_tng_reader_and_zip_source(corpus_config: pipeline.CorpusConfig):
    corpus_reader = tng.CorpusReader(
        source=tng.ZipSource(source_path=corpus_config.pipeline_payload.source),
        reader_opts=corpus_config.text_reader_opts,
        preprocess=None,
        tokenizer=None,
    )
    assert corpus_reader is not None
    names = [x[0] for x in corpus_reader]

    assert names == ['prot_197677__25.csv', 'prot_197677__26.csv', 'prot_197677__27.csv']


@pytest.mark.long_running
def test_sparv_tagged_frame_to_tokens(corpus_config: pipeline.CorpusConfig):
    p = pipeline.CorpusPipeline(config=corpus_config)
    corpus_config.serialize_opts.feather_folder = None
    load = tasks.LoadTaggedCSV(
        filename=corpus_config.pipeline_payload.source,
        serialize_opts=corpus_config.serialize_opts,
        extra_reader_opts=corpus_config.text_reader_opts,
    )

    tagged_columns: dict = corpus_config.pipeline_payload.tagged_columns_names
    extract = tagged_frame.TaggedFrameToTokens(
        extract_opts=ExtractTaggedTokensOpts(lemmatize=True, **tagged_columns, filter_opts=dict(is_punct=False)),
    )

    p.add([load, extract])

    payloads = [p for p in p.resolve()]

    assert [x.document_name for x in payloads] == ['prot_197677__25', 'prot_197677__26', 'prot_197677__27']
    assert all(x.content_type == pipeline.ContentType.TOKENS for x in payloads)
    assert all(isinstance(x.content, list) for x in payloads)
    assert len(payloads) == 3
