import pytest

import penelope.corpus.readers as readers
from tests.utils import TEST_CORPUS_FILENAME

# pylint: disable=too-many-arguments


def test_streamify_text_source_smoke_test():
    stream = readers.ZipCorpusReader(
        TEST_CORPUS_FILENAME,
        reader_opts=readers.TextReaderOpts(filename_pattern="*.txt", filename_filter=None, as_binary=False),
    )

    filename, text = next(stream)

    assert filename == 'dikt_2019_01_test.txt'
    assert (
        text
        == 'Tre svarta ekar ur snön.\r\nSå grova, men fingerfärdiga.\r\nUr deras väldiga flaskor\r\nska grönskan skumma i vår.'
    )


@pytest.mark.xfail
def test_streamify_text_source_smoke_test_raises_exception():
    stream = readers.ZipCorpusReader(
        TEST_CORPUS_FILENAME,
        reader_opts=readers.TextReaderOpts(filename_pattern="*.dat", filename_filter=None, as_binary=False),
    )

    with pytest.raises(StopIteration):
        _, _ = next(stream)
