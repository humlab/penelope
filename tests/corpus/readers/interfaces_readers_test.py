import pandas as pd

import penelope.corpus.readers as readers
from penelope.corpus.readers import ICorpusReader, TextReaderOpts
from tests.sparv.sparv_csv_iterator_test import SPARV_CSV_EXPORT_FILENAME_SMALL
from tests.sparv.sparv_xml_iterator_test import SPARV_XML_EXPORT_FILENAME_SMALL
from tests.utils import TEST_CORPUS_FILENAME


def test_text_tokenizer_interface():
    assert issubclass(readers.TokenizeTextReader, ICorpusReader)

    instance = readers.TokenizeTextReader(source=["a b c", "e f g"])
    assert isinstance(instance, ICorpusReader)

    # assert [('document_1.txt', ['a', 'b', 'c']), ('document_2.txt', ['e', 'f', 'g'])] == [ x for x in corpus ]


def test_pandas_reader_interface():
    assert issubclass(readers.PandasCorpusReader, ICorpusReader)

    source = pd.DataFrame(data={'filename': [], 'txt': []})
    instance = readers.PandasCorpusReader(source)
    assert isinstance(instance, ICorpusReader)


def test_sparv_csv_tokenizer_interface():
    assert issubclass(readers.SparvCsvReader, ICorpusReader)

    instance = readers.SparvCsvReader(SPARV_CSV_EXPORT_FILENAME_SMALL, reader_opts=TextReaderOpts())
    assert isinstance(instance, ICorpusReader)


def test_sparv_xml3_tokenizer_interface():
    assert issubclass(readers.Sparv3XmlReader, ICorpusReader)

    instance = readers.Sparv3XmlReader('./tests/test_data/sparv_data/sou_test_sparv3_xml.zip')
    assert isinstance(instance, ICorpusReader)


def test_sparv_xml_tokenizer_interface():
    assert issubclass(readers.SparvXmlReader, ICorpusReader)

    instance = readers.SparvXmlReader(SPARV_XML_EXPORT_FILENAME_SMALL)
    assert isinstance(instance, ICorpusReader)


def test_zip_text_iterator_interface():
    assert issubclass(readers.ZipCorpusReader, ICorpusReader)
    instance = readers.ZipCorpusReader(TEST_CORPUS_FILENAME, reader_opts=TextReaderOpts())
    assert isinstance(instance, ICorpusReader)
