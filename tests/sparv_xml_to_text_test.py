import pytest # pylint: disable=unused-import
import lxml

import penelope.corpus.sparv.sparv_xml_to_text as sparv

SPARV_XML_EXPORT_FILENAME = './penelope/tests/test_data/sparv_xml_export_small.xml'

def sparv_xml_test_file():
    with open(SPARV_XML_EXPORT_FILENAME, "rb") as fp:
        return fp.read()

def test_extract_call_with_no_postag_should_fail():

    content = sparv_xml_test_file()
    parser = sparv.SparvXml2Text()

    with pytest.raises(lxml.etree.XSLTApplyError):
        _ = parser.transform(content)

def test_extract_when_no_filter_or_lemmatize_returns_original_text():

    expected = "Rödräven är ett hunddjur som har en mycket vidsträckt utbredning över norra halvklotet . "
    content = sparv_xml_test_file()
    parser = sparv.SparvXml2Text(pos_includes="", lemmatize=False, delimiter=" ", append_pos="", pos_excludes="")

    result = parser.transform(content)

    assert result == expected

def test_extract_when_ignore_punctuation_filters_out_punctuations():

    expected = "Rödräven är ett hunddjur som har en mycket vidsträckt utbredning över norra halvklotet "
    content = sparv_xml_test_file()
    parser = sparv.SparvXml2Text(pos_includes="", lemmatize=False, delimiter=" ", append_pos="", pos_excludes="|MAD|MID|PAD|")

    result = parser.transform(content)

    assert result == expected

def test_extract_when_lemmatized_returns_baseform():

    expected = 'rödräv vara en hunddjur som ha en mycken vidsträckt utbredning över norra halvklot . '
    content = sparv_xml_test_file()
    parser = sparv.SparvXml2Text(pos_includes="", lemmatize=True, delimiter=" ", append_pos="", pos_excludes="")

    result = parser.transform(content)

    assert result == expected

def test_extract_when_lemmatized_and_filter_nouns_returns_nouns_in_baseform():

    expected = 'rödräv hunddjur utbredning halvklot '
    content = sparv_xml_test_file()
    parser = sparv.SparvXml2Text(pos_includes="|NN|", lemmatize=True, delimiter=" ", append_pos="", pos_excludes="|MAD|MID|PAD|")

    result = parser.transform(content)

    assert result == expected

def test_extract_when_lemmatized_and_filter_nouns_returns_nouns_in_baseform_with_given_delimiter():

    expected = 'rödräv|hunddjur|utbredning|halvklot|'
    content = sparv_xml_test_file()
    parser = sparv.SparvXml2Text(pos_includes="|NN|", lemmatize=True, delimiter="|", append_pos="", pos_excludes="|MAD|MID|PAD|")

    result = parser.transform(content)

    assert result == expected
