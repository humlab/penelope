import penelope.corpus.readers as readers
import pytest  # pylint: disable=unused-import

SPARV_XML_EXPORT_FILENAME = './tests/test_data/sparv_xml_export.xml'
SPARV_XML_EXPORT_FILENAME_SMALL = './tests/test_data/sparv_xml_export_small.xml'
SPARV_ZIPPED_XML_EXPORT_FILENAME = './tests/test_data/sparv_zipped_xml_export.zip'


def sparv_xml_test_file():
    with open(SPARV_XML_EXPORT_FILENAME, "rb") as fp:
        return fp.read()


DEFAULT_OPTS = dict(
    pos_includes='',
    lemmatize=True,
    chunk_size=None,
    xslt_filename=None,
    delimiter="|",
    append_pos="",
    pos_excludes="|MAD|MID|PAD|",
)


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

    opts = dict(pos_includes='', lemmatize=False, chunk_size=None, pos_excludes="")

    reader = readers.SparvXmlTokenizer(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == document_name


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

    opts = dict(pos_includes='', lemmatize=True, chunk_size=None, pos_excludes="")

    reader = readers.SparvXmlTokenizer(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == document_name


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

    opts = dict(pos_includes='', lemmatize=True, chunk_size=None, pos_excludes="|MAD|MID|PAD|")

    reader = readers.SparvXmlTokenizer(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == tokens
    assert expected_name == document_name


def test_reader_when_only_nouns_ignore_puncts_returns_filter_outs_puncts():

    expected = ['rödräv', 'hunddjur', 'utbredning', 'halvklot']
    expected_name = "sparv_xml_export_small.txt"

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    reader = readers.SparvXmlTokenizer(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == document_name


def test_reader_when_chunk_size_specified_returns_chunked_text():

    expected_documents = [['rödräv', 'hunddjur'], ['utbredning', 'halvklot']]
    expected_names = ["sparv_xml_export_small_001.txt", "sparv_xml_export_small_002.txt"]

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=2)

    reader = readers.SparvXmlTokenizer(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    for i, (document_name, tokens) in enumerate(reader):

        assert expected_documents[i] == list(tokens)
        assert expected_names[i] == document_name


def test_reader_when_source_is_zipped_archive_succeeds():

    expected_documents = [
        ['rödräv', 'hunddjur', 'utbredning', 'halvklot'],
        ['fjällräv', 'fjällvärld', 'liv', 'fjällräv', 'vinter', 'men', 'variant', 'år'],
    ]
    expected_names = ["document_001.txt", "document_002.txt"]

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    reader = readers.SparvXmlTokenizer(SPARV_ZIPPED_XML_EXPORT_FILENAME, **opts)

    for i, (document_name, tokens) in enumerate(reader):

        assert expected_documents[i] == list(tokens)
        assert expected_names[i] == document_name


def test_reader_when_source_is_sparv3_succeeds():

    sparv_zipped_xml_export_v3_filename = './tests/test_data/sou_test_sparv3_xml.zip'

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)  # , xslt_filename=xslt_filename)

    reader = readers.Sparv3XmlTokenizer(sparv_zipped_xml_export_v3_filename, **opts)

    for _, (_, tokens) in enumerate(reader):

        assert len(list(tokens)) > 0
