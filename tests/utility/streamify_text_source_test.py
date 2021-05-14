import os
import pytest
import uuid
from penelope.utility import list_any_source, streamify_any_source, unpack
from tests.utils import TEST_CORPUS_FILENAME
import shutil

# pylint: disable=too-many-arguments
EXPECTED_TEXT_FILES = [
    'dikt_2019_01_test.txt',
    'dikt_2019_02_test.txt',
    'dikt_2019_03_test.txt',
    'dikt_2020_01_test.txt',
    'dikt_2020_02_test.txt',
]


def test_streamify_any_source_smoke_test():
    stream = streamify_any_source(TEST_CORPUS_FILENAME)
    filenames = [x[0] for x in stream]
    assert filenames == ['README.md'] + EXPECTED_TEXT_FILES
    stream = streamify_any_source(TEST_CORPUS_FILENAME, filename_pattern="*.md")
    filenames = [x[0] for x in stream]
    assert filenames == ['README.md']

    stream = streamify_any_source(TEST_CORPUS_FILENAME)
    filenames = [x[0] for x in stream]
    assert filenames == ['README.md'] + EXPECTED_TEXT_FILES

    stream = streamify_any_source(TEST_CORPUS_FILENAME, filename_filter=lambda x: x.endswith(".txt"))
    filenames = [x[0] for x in stream]
    assert filenames == EXPECTED_TEXT_FILES

    stream = streamify_any_source(TEST_CORPUS_FILENAME, filename_filter=EXPECTED_TEXT_FILES)
    filenames = [x[0] for x in stream]
    assert filenames == EXPECTED_TEXT_FILES


def test_next_of_streamified_zipped_source_returns_document():

    folder: str = os.path.join('./tests/output', str(uuid.uuid1()))
    os.makedirs(folder)

    unpack(TEST_CORPUS_FILENAME, folder, create_sub_folder=False)

    stream = streamify_any_source(folder)

    filenames = [x[0] for x in stream]
    assert filenames == ['README.md'] + EXPECTED_TEXT_FILES

    shutil.rmtree(folder, ignore_errors=True)


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
