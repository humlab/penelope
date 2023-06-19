import io

import pandas as pd
import pytest

import penelope.corpus.sparv.sparv_xml_to_csv as sparv
from penelope.corpus import TextReaderOpts
from penelope.corpus.readers import tng

SPARV_XML_EXPORT_FILENAME = './tests/test_data/sparv_data/sparv_xml_export_small.xml'


def sparv_xml_test_file():
    with open(SPARV_XML_EXPORT_FILENAME, "rb") as fp:
        return fp.read()


def test_extract_to_tsv():
    expected = [
        'Rödräven\trödräv\tNN',
        'är\tvara\tVB',
        'ett\ten\tDT',
        'hunddjur\thunddjur\tNN',
        'som\t\tKN',
        'har\tha\tVB',
        'en\ten\tAB',
        'mycket\tmycken\tPN',
        'vidsträckt\tvidsträckt\tJJ',
        'utbredning\tutbredning\tNN',
        'över\töver\tPP',
        'norra\tnorra\tJJ',
        'halvklotet\thalvklot\tNN',
        '.\t\tMAD',
    ]
    content = sparv_xml_test_file()
    parser = sparv.SparvXml2CSV(delimiter="\t", version=4)

    result = parser.transform(content)

    assert result == '\r'.join(expected) + '\r'


@pytest.mark.long_running
@pytest.mark.skip("deprecated Sparv v3")
def test_xml_to_csv_corpus_reader():
    source_path = './tests/test_data/sparv_data/sou_test_sparv3_xml.zip'
    parser = sparv.SparvXml2CSV(delimiter="\t", version=3)
    source = tng.ZipSource(source_path=source_path)
    reader_opts = TextReaderOpts(filename_pattern='*.xml', as_binary=True)

    with tng.ZipSource(source_path=source_path) as source:
        data = [source.read(x, as_binary=True) for x in source.namelist()]

    assert len(data) > 0

    def xml_to_csv(xml_doc: bytes):
        csv_doc = parser.transform(xml_doc)
        return csv_doc

    reader = tng.CorpusReader(source=source, reader_opts=reader_opts, preprocess=xml_to_csv)

    data = [x for x in reader]

    assert len(data) > 0

    def xml_to_tagged_frame(xml_doc: bytes) -> pd.DataFrame:
        csv_doc = parser.transform(xml_doc)
        df = pd.read_csv(io.StringIO(csv_doc), sep='\t', quoting=3)
        df.columns = ['text', 'lemma', 'pos']
        return df

    reader = tng.CorpusReader(source=source, reader_opts=reader_opts, preprocess=xml_to_tagged_frame)

    data = [x for x in reader]

    assert len(data) > 0
    assert all(isinstance(x[1], pd.DataFrame) for x in data)


def test_create_sparv_xml_corpus_reader():
    filename = './tests/test_data/sparv_data/sou_sparv3_3files_xml.zip'
    reader_opts = TextReaderOpts(filename_pattern='*.xml', as_binary=True, sep='\t')
    reader = tng.create_sparv_xml_corpus_reader(
        source_path=filename,
        reader_opts=reader_opts,
        sparv_version=3,
        content_type="pandas",
    )
    data = [x for x in reader]

    assert len(data) > 0
    assert all(isinstance(x[1], pd.DataFrame) for x in data)
