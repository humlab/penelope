import pandas as pd
import penelope.corpus.readers as readers
from penelope.corpus.readers import ICorpusReader
from penelope.corpus.readers.interfaces import TextReaderOpts

from .sparv_csv_iterator_test import SPARV_CSV_EXPORT_FILENAME_SMALL
from .sparv_xml_iterator_test import SPARV_XML_EXPORT_FILENAME_SMALL
from .utils import TEST_CORPUS_FILENAME


def test_text_tokenizer_interface():

    assert issubclass(readers.TextTokenizer, ICorpusReader)

    instance = readers.TextTokenizer(source=["a b c", "e f g"])
    assert isinstance(instance, ICorpusReader)

    # assert [('document_1.txt', ['a', 'b', 'c']), ('document_2.txt', ['e', 'f', 'g'])] == [ x for x in corpus ]


def test_dataframe_text_tokenizer_interface():

    assert issubclass(readers.DataFrameTextTokenizer, ICorpusReader)

    source = pd.DataFrame(data={'filename': [], 'txt': []})
    instance = readers.DataFrameTextTokenizer(source)
    assert isinstance(instance, ICorpusReader)


def test_sparv_csv_tokenizer_interface():

    assert issubclass(readers.SparvCsvTokenizer, ICorpusReader)

    instance = readers.SparvCsvTokenizer(SPARV_CSV_EXPORT_FILENAME_SMALL, reader_opts=TextReaderOpts())
    assert isinstance(instance, ICorpusReader)


def test_sparv_xml3_tokenizer_interface():

    assert issubclass(readers.Sparv3XmlTokenizer, ICorpusReader)

    instance = readers.Sparv3XmlTokenizer('./tests/test_data/sou_test_sparv3_xml.zip')
    assert isinstance(instance, ICorpusReader)


def test_sparv_xml_tokenizer_interface():

    assert issubclass(readers.SparvXmlTokenizer, ICorpusReader)

    instance = readers.SparvXmlTokenizer(SPARV_XML_EXPORT_FILENAME_SMALL)
    assert isinstance(instance, ICorpusReader)


def test_zip_text_iterator_interface():

    assert issubclass(readers.ZipTextIterator, ICorpusReader)
    instance = readers.ZipTextIterator(TEST_CORPUS_FILENAME, reader_opts=TextReaderOpts())
    assert isinstance(instance, ICorpusReader)
