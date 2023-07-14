import pandas as pd

from penelope import corpus as pc
from penelope.corpus import sparv
from tests.sparv.sparv_csv_iterator_test import SPARV_CSV_EXPORT_FILENAME_SMALL
from tests.sparv.sparv_xml_iterator_test import SPARV_XML_EXPORT_FILENAME_SMALL
from tests.utils import TEST_CORPUS_FILENAME


def test_text_tokenizer_interface():
    assert issubclass(pc.TokenizeTextReader, pc.ICorpusReader)

    instance = pc.TokenizeTextReader(source=["a b c", "e f g"])
    assert isinstance(instance, pc.ICorpusReader)

    # assert [('document_1.txt', ['a', 'b', 'c']), ('document_2.txt', ['e', 'f', 'g'])] == [ x for x in corpus ]


def test_pandas_reader_interface():
    assert issubclass(pc.PandasCorpusReader, pc.ICorpusReader)

    source = pd.DataFrame(data={'filename': [], 'txt': []})
    instance = pc.PandasCorpusReader(source)
    assert isinstance(instance, pc.ICorpusReader)


def test_sparv_csv_tokenizer_interface():
    assert issubclass(sparv.SparvCsvReader, pc.ICorpusReader)

    instance = sparv.SparvCsvReader(SPARV_CSV_EXPORT_FILENAME_SMALL, reader_opts=pc.TextReaderOpts())
    assert isinstance(instance, pc.ICorpusReader)


def test_sparv_xml3_tokenizer_interface():
    assert issubclass(sparv.Sparv3XmlReader, pc.ICorpusReader)

    instance = sparv.Sparv3XmlReader('./tests/test_data/sparv_data/sou_test_sparv3_xml.zip')
    assert isinstance(instance, pc.ICorpusReader)


def test_sparv_xml_tokenizer_interface():
    assert issubclass(sparv.SparvXmlReader, pc.ICorpusReader)

    instance = sparv.SparvXmlReader(SPARV_XML_EXPORT_FILENAME_SMALL)
    assert isinstance(instance, pc.ICorpusReader)


def test_zip_text_iterator_interface():
    assert issubclass(pc.ZipCorpusReader, pc.ICorpusReader)
    instance = pc.ZipCorpusReader(TEST_CORPUS_FILENAME, reader_opts=pc.TextReaderOpts())
    assert isinstance(instance, pc.ICorpusReader)
