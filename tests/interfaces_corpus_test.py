import pytest

import penelope.corpus.readers as readers
from penelope.corpus import (
    ICorpus,
    ITokenizedCorpus,
    SegmentedTextCorpus,
    SimpleTextLinesCorpus,
    SparvTokenizedCsvCorpus,
    SparvTokenizedXmlCorpus,
    TokenizedCorpus,
)
from tests.sparv_xml_corpus_test import SPARV_XML_EXPORT_FILENAME_SMALL


def test_corpus_interface_subclassed():
    class TestCorpus(ICorpus):
        def __next__(self):
            'Return the next item from the iterator. When exhausted, raise StopIteration'
            raise StopIteration

        def __iter__(self):
            return self

        @property
        def metadata(self):
            return None

        @property
        def filenames(self):
            return None

    corpus = TestCorpus()

    assert isinstance(corpus, ICorpus)


def test_tokenized_corpus_interface():

    assert issubclass(TokenizedCorpus, ITokenizedCorpus)

    source = readers.TextTokenizer(source_path=["a b c", "e f g"])
    instance = TokenizedCorpus(source)
    assert isinstance(instance, ITokenizedCorpus)

    # assert [('document_1.txt', ['a', 'b', 'c']), ('document_2.txt', ['e', 'f', 'g'])] == [ x for x in corpus ]


def test_segmented_stokenized_corpus_interface():

    assert issubclass(SegmentedTextCorpus, ITokenizedCorpus)

    source = readers.TextTokenizer(source_path=["a b c", "e f g"])
    instance = SegmentedTextCorpus(source, segment_strategy="sentence")
    assert isinstance(instance, ITokenizedCorpus)


def test_sparv_xml_corpus_interface():

    assert issubclass(SparvTokenizedXmlCorpus, ITokenizedCorpus)

    instance = SparvTokenizedXmlCorpus(SPARV_XML_EXPORT_FILENAME_SMALL, version=4)
    assert isinstance(instance, ITokenizedCorpus)


def test_sparv3_xml_corpus_interface():

    assert issubclass(SparvTokenizedXmlCorpus, ITokenizedCorpus)

    instance = SparvTokenizedXmlCorpus('./tests/test_data/sou_test_sparv3_xml.zip', version=3)
    assert isinstance(instance, ITokenizedCorpus)


def test_sparv3_csv_corpus_interface():

    assert issubclass(SparvTokenizedCsvCorpus, ITokenizedCorpus)

    instance = SparvTokenizedCsvCorpus('./tests/test_data/sparv_zipped_csv_export.zip')
    assert isinstance(instance, ITokenizedCorpus)


def test_text_lines_corpus_interface():

    assert issubclass(SimpleTextLinesCorpus, ITokenizedCorpus)

    instance = SimpleTextLinesCorpus(
        filename='./tests/test_data/tranströmer.txt',
        fields={'filename': 0, 'title': 1, 'text': 2},
        meta_fields=["year:_:1", "year_serial_id:_:2"],
    )
    assert isinstance(instance, ITokenizedCorpus)
