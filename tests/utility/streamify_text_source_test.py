import pytest
from penelope.utility import list_any_source, streamify_any_source
from tests.utils import TEST_CORPUS_FILENAME

# pylint: disable=too-many-arguments


def test_streamify_any_source_smoke_test():
    expected_text_files = [
        'dikt_2019_01_test.txt',
        'dikt_2019_02_test.txt',
        'dikt_2019_03_test.txt',
        'dikt_2020_01_test.txt',
        'dikt_2020_02_test.txt',
    ]
    stream = streamify_any_source(TEST_CORPUS_FILENAME)
    filenames = [x[0] for x in stream]
    assert filenames == ['README.md'] + expected_text_files
    stream = streamify_any_source(TEST_CORPUS_FILENAME, filename_pattern="*.md")
    filenames = [x[0] for x in stream]
    assert filenames == ['README.md']

    stream = streamify_any_source(TEST_CORPUS_FILENAME)
    filenames = [x[0] for x in stream]
    assert filenames == ['README.md'] + expected_text_files

    stream = streamify_any_source(TEST_CORPUS_FILENAME, filename_filter=lambda x: x.endswith(".txt"))
    filenames = [x[0] for x in stream]
    assert filenames == expected_text_files

    stream = streamify_any_source(TEST_CORPUS_FILENAME, filename_filter=expected_text_files)
    filenames = [x[0] for x in stream]
    assert filenames == expected_text_files


def test_next_of_streamified_zipped_source_returns_document():

    stream = streamify_any_source(TEST_CORPUS_FILENAME)

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


def test_create_iterator():
    stream = streamify_any_source(
        TEST_CORPUS_FILENAME, ['dikt_2019_01_test.txt'], filename_pattern='*.txt', as_binary=False
    )
    assert len([x for x in stream]) == 1


def test_list_filenames_when_source_is_a_filename():

    filenames = list_any_source(TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=None)
    assert len(filenames) == 5

    filenames = list_any_source(TEST_CORPUS_FILENAME, filename_pattern='*.dat', filename_filter=None)
    assert len(filenames) == 0

    filenames = list_any_source(
        TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=['dikt_2019_01_test.txt']
    )
    assert len(filenames) == 1

    filenames = list_any_source(
        TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=lambda x: x == 'dikt_2019_01_test.txt'
    )
    assert len(filenames) == 1


def list_filenames_when_source_is_folder():
    # utility.list_filenames(folder_or_zip, filename_pattern: str="*.txt", filename_filter: Union[List[str],Callable]=None)
    pass
