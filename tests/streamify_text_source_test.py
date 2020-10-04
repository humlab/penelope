import pytest

from penelope.corpus.readers.streamify_text_source import streamify_text_source

from tests.utils import TEST_CORPUS_FILENAME

# pylint: disable=too-many-arguments

def test_streamify_text_source_smoke_test():

    stream = streamify_text_source(TEST_CORPUS_FILENAME)

    assert stream is not None

def test_next_of_streamified_zipped_source_returns_document_strem():

    stream = streamify_text_source(TEST_CORPUS_FILENAME)

    assert stream is not None
    assert next(stream) is not None

@pytest.mark.xfail
def test_next_of_streamified_folder_source_returns_document_stream():
    assert False

@pytest.mark.xfail
def test_next_of_streamified_already_stream_source_returns_document_stream():
    assert False

@pytest.mark.xfail
def test_next_of_streamified_of_text_chunk_returns_single_document():
    assert False

@pytest.mark.xfail
def test_next_of_streamified_when_not_str_nor_stream_should_fail():
    assert False

# NOTE: Test pattern, txt/xml, filename_filter (list/function), as_binary
