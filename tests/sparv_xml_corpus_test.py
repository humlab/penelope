import os
import uuid

import pytest  # pylint: disable=unused-import

import penelope.corpus.sparv_corpus as sparv_corpus
from penelope.utility import read_from_archive

from .utils import OUTPUT_FOLDER

SPARV_XML_EXPORT_FILENAME = './tests/test_data/sparv_xml_export.xml'
SPARV_XML_EXPORT_FILENAME_SMALL = './tests/test_data/sparv_xml_export_small.xml'
SPARV_ZIPPED_XML_EXPORT_FILENAME = './tests/test_data/sparv_zipped_xml_export.zip'
SPARV3_ZIPPED_XML_EXPORT_FILENAME = './tests/test_data/sou_test_sparv3_xml.zip'


def test_reader_store_result():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    expected_documents = [
        ['rödräv', 'hunddjur', 'utbredning', 'halvklot'],
        ['fjällräv', 'fjällvärld', 'liv', 'fjällräv', 'vinter', 'men', 'variant', 'år'],
    ]
    expected_names = ["document_001.txt", "document_002.txt"]

    target_filename = os.path.join(OUTPUT_FOLDER, 'test_reader_store_result.zip')

    sparv_corpus.sparv_xml_extract_and_store(
        SPARV_ZIPPED_XML_EXPORT_FILENAME,
        target_filename,
        version=4,
        pos_includes='|NN|',
        lemmatize=True,
        tokens_transform_opts=dict(to_lower=True),
    )

    for i in range(0, len(expected_names)):

        content = read_from_archive(target_filename, expected_names[i], as_binary=False)

        assert ' '.join(expected_documents[i]) == content

    os.remove(target_filename)


def test_sparv_extract_and_store_when_only_nouns_and_source_is_sparv3_succeeds():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    target_filename = os.path.join(OUTPUT_FOLDER, f'{uuid.uuid1()}.zip')

    sparv_corpus.sparv_xml_extract_and_store(
        SPARV3_ZIPPED_XML_EXPORT_FILENAME,
        target_filename,
        version=3,
        pos_includes='|NN|',
        lemmatize=False,
        tokens_transform_opts={
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
        },
    )

    expected_document_start = "utredningar justitiedepartementet förslag utlänningslag angående om- händertagande förläggning års gere ide to lm \rstatens utredningar förteckning betänkande förslag utlänningslag lag omhändertagande utlänning anstalt förläggning tryckort tryckorten bokstäverna fetstil begynnelse- bokstäverna departement"

    test_filename = "sou_1945_1.txt"

    content = read_from_archive(target_filename, test_filename, as_binary=False)

    assert content.startswith(expected_document_start)

    os.remove(target_filename)
