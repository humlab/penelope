import os

import pytest  # pylint: disable=unused-import

import penelope.utility.file_utility as file_utility
import penelope.corpus.sparv_corpus as sparv_corpus
import penelope.corpus.readers.sparv_xml_tokenizer as sparv_reader

SPARV_XML_EXPORT_FILENAME = './penelope/tests/test_data/sparv_xml_export.xml'
SPARV_XML_EXPORT_FILENAME_SMALL = './penelope/tests/test_data/sparv_xml_export_small.xml'
SPARV_ZIPPED_XML_EXPORT_FILENAME = './penelope/tests/test_data/sparv_zipped_xml_export.zip'
SPARV3_ZIPPED_XML_EXPORT_FILENAME = './penelope/tests/test_data/sou_test_sparv3_xml.zip'


def test_reader_store_result():

    expected_documents = [
        ['rödräv', 'hunddjur', 'utbredning', 'halvklot'],
        ['fjällräv', 'fjällvärld', 'liv', 'fjällräv', 'vinter', 'men', 'variant', 'år'],
    ]
    expected_names = ["document_001.txt", "document_002.txt"]

    target_filename = './penelope/tests/output/sparv_extract_and_store.zip'

    opts = dict(
        version=4,
        pos_includes='|NN|',
        lemmatize=True,
        chunk_size=None,
        to_lower=True,
    )

    sparv_corpus.sparv_extract_and_store(SPARV_ZIPPED_XML_EXPORT_FILENAME, target_filename, **opts)

    for i in range(0, len(expected_names)):

        content = file_utility.read(target_filename, expected_names[i], as_binary=False)

        assert ' '.join(expected_documents[i]) == content


def test_sparv_extract_and_store_when_only_nouns_and_source_is_sparv3_succeeds():

    os.makedirs('./penelope/tests/output', exist_ok=True)

    opts = {
        'pos_includes': '|NN|',
        'lemmatize': False,
        'chunk_size': None,
        'version': 3,
        'to_lower': True,
        'min_len': 2,
        'stopwords': ['<text>']
        # only_alphabetic: bool=False,
        # only_any_alphanumeric: bool=False,
        # to_lower: bool = False,
        # to_upper: bool = False,
        # min_len: int = None,
        # max_len: int = None,
        # remove_accents: bool = False,
        # remove_stopwords: bool = False,
        # stopwords: Any = None,
        # extra_stopwords: List[str] = None,
        # language: str = "swedish",
        # keep_numerals: bool = True,
        # keep_symbols: bool = True
    }

    target_filename = './penelope/tests/output/sou_test_sparv3_extracted_txt.zip'

    sparv_corpus.sparv_extract_and_store(SPARV3_ZIPPED_XML_EXPORT_FILENAME, target_filename, **opts)

    expected_document_start = "utredningar justitiedepartementet förslag utlänningslag angående om- händertagande förläggning års gere ide to lm \rstatens utredningar förteckning betänkande förslag utlänningslag lag omhändertagande utlänning anstalt förläggning tryckort tryckorten bokstäverna fetstil begynnelse- bokstäverna departement"

    test_filename = "sou_1945_1.txt"

    content = file_utility.read(target_filename, test_filename, as_binary=False)

    assert content.startswith(expected_document_start)


def test_corpus_when_source_is_sparv3_succeeds():

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    reader = sparv_reader.Sparv3XmlTokenizer(SPARV3_ZIPPED_XML_EXPORT_FILENAME, **opts)

    for _, (_, tokens) in enumerate(reader):

        assert len(list(tokens)) > 0

    # id2word = gensim.corpora.Dictionary(terms)
    # corpus = [ id2word.doc2bow(tokens) for tokens in terms ]
