import penelope.corpus.readers as readers
import pytest
from penelope.corpus.readers import ExtractTaggedTokensOpts

SPARV_XML_EXPORT_FILENAME = './tests/test_data/sparv_xml_export.xml'
SPARV_XML_EXPORT_FILENAME_SMALL = './tests/test_data/sparv_xml_export_small.xml'
SPARV_ZIPPED_XML_EXPORT_FILENAME = './tests/test_data/sparv_zipped_xml_export.zip'


def sparv_xml_test_file():
    with open(SPARV_XML_EXPORT_FILENAME, "rb") as fp:
        return fp.read()


def test_reader_when_no_transforms_returns_source_tokens():

    expected = [
        'Rödräven',
        'är',
        'ett',
        'hunddjur',
        'som',
        'har',
        'en',
        'mycket',
        'vidsträckt',
        'utbredning',
        'över',
        'norra',
        'halvklotet',
        '.',
    ]
    expected_name = "sparv_xml_export_small.txt"

    reader = readers.SparvXmlReader(
        SPARV_XML_EXPORT_FILENAME_SMALL,
        chunk_size=None,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='', lemmatize=False, pos_excludes=None),
    )

    filename, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == filename


def test_reader_when_lemmatized_returns_tokens_in_baseform():

    expected = [
        'rödräv',
        'vara',
        'en',
        'hunddjur',
        'som',
        'ha',
        'en',
        'mycken',
        'vidsträckt',
        'utbredning',
        'över',
        'norra',
        'halvklot',
        '.',
    ]
    expected_name = "sparv_xml_export_small.txt"

    reader = readers.SparvXmlReader(
        SPARV_XML_EXPORT_FILENAME_SMALL,
        chunk_size=None,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='', lemmatize=True, pos_excludes=None),
    )

    filename, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == filename


def test_reader_when_ignore_puncts_returns_filter_outs_puncts():

    expected = [
        'rödräv',
        'vara',
        'en',
        'hunddjur',
        'som',
        'ha',
        'en',
        'mycken',
        'vidsträckt',
        'utbredning',
        'över',
        'norra',
        'halvklot',
    ]
    expected_name = "sparv_xml_export_small.txt"

    reader = readers.SparvXmlReader(
        SPARV_XML_EXPORT_FILENAME_SMALL,
        chunk_size=None,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='', lemmatize=True, pos_excludes="|MAD|MID|PAD|"),
    )

    filename, tokens = next(iter(reader))

    assert expected == tokens
    assert expected_name == filename


def test_reader_when_only_nouns_ignore_puncts_returns_filter_outs_puncts():

    expected = ['rödräv', 'hunddjur', 'utbredning', 'halvklot']
    expected_name = "sparv_xml_export_small.txt"

    reader = readers.SparvXmlReader(
        SPARV_XML_EXPORT_FILENAME_SMALL,
        chunk_size=None,
        extract_tokens_opts=ExtractTaggedTokensOpts(
            pos_includes='|NN|',
            lemmatize=True,
        ),
    )

    filename, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == filename


def test_reader_when_chunk_size_specified_returns_chunked_text():

    expected_documents = [['rödräv', 'hunddjur'], ['utbredning', 'halvklot']]
    expected_names = ["sparv_xml_export_small_001.txt", "sparv_xml_export_small_002.txt"]

    reader = readers.SparvXmlReader(
        SPARV_XML_EXPORT_FILENAME_SMALL,
        chunk_size=2,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|', lemmatize=True),
    )

    for i, (filename, tokens) in enumerate(reader):

        assert expected_documents[i] == list(tokens)
        assert expected_names[i] == filename


def test_reader_when_source_is_zipped_archive_succeeds():

    expected_documents = [
        ['rödräv', 'hunddjur', 'utbredning', 'halvklot'],
        ['fjällräv', 'fjällvärld', 'liv', 'fjällräv', 'vinter', 'men', 'variant', 'år'],
    ]
    expected_names = ["document_001.txt", "document_002.txt"]

    reader = readers.SparvXmlReader(
        SPARV_ZIPPED_XML_EXPORT_FILENAME,
        chunk_size=None,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|', lemmatize=True),
    )

    for i, (filename, tokens) in enumerate(reader):

        assert expected_documents[i] == list(tokens)
        assert expected_names[i] == filename


@pytest.mark.skip('Long running')
def test_reader_when_source_is_sparv3_succeeds():

    sparv_zipped_xml_export_v3_filename = './tests/test_data/sou_test_sparv3_xml.zip'

    reader = readers.Sparv3XmlReader(
        sparv_zipped_xml_export_v3_filename,
        chunk_size=None,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|', lemmatize=True),
    )

    for _, (_, tokens) in enumerate(reader):

        assert len(list(tokens)) > 0
