import os
import zipfile
from fnmatch import fnmatch
from typing import Any, List, Optional

import pandas as pd
import pytest

import penelope.corpus.readers.tng.reader as cr
import penelope.corpus.readers.tng.sources as cs
import penelope.corpus.transformer as tt
from penelope.corpus.readers.interfaces import TextReaderOpts
from penelope.utility import streamify_zip_source, strip_path_and_extension, strip_paths, zip_utils
from tests.utils import OUTPUT_FOLDER, TEST_CORPUS_FILENAME

EXPECTED_TEXT_FILES = [
    'dikt_2019_01_test.txt',
    'dikt_2019_02_test.txt',
    'dikt_2019_03_test.txt',
    'dikt_2020_01_test.txt',
    'dikt_2020_02_test.txt',
]


def test_zip_wrapper():
    filename = './tests/test_data/SSI/legal_instrument_five_docs_test.zip'
    with zipfile.ZipFile(filename, 'r') as zf:
        names = zf.namelist()
        names_zip = zip_utils.list_filenames(zip_or_filename=zf)
    names_str = zip_utils.list_filenames(zip_or_filename=filename)
    assert names == names_zip == names_str


def test_transformer():
    transform_opts = tt.TextTransformOpts(transforms="")

    transform_opts += "strip-accents"
    assert transform_opts.transforms == "strip-accents"

    transform_opts -= "strip-accents"
    assert transform_opts.transforms == ""

    transform_opts = tt.TextTransformOpts(transforms="")
    transform_opts += "strip-accents"

    transformer = tt.TextTransformer(transform_opts=transform_opts)
    result = transformer.transform("Rågér")
    assert result == "Rager"

    transformer.transform_opts -= "strip-accents"
    result = transformer.transform("Rågér")
    assert result == "Rågér"

    transformer.transform_opts.clear()
    transformer.transform_opts += "dehyphen"
    result = transformer.transform("mål-\nvakt")
    assert result.strip() == "målvakt"

    transformer.transform_opts.clear()
    transformer.transform_opts += "normalize-whitespace"
    result = transformer.transform("mål    vakt")
    assert result == "mål vakt"


def create_test_source_info(filenames: List[str]) -> cr.SourceInfo:
    basenames = [strip_paths(x) for x in filenames]
    return cr.SourceInfo(
        names=basenames,
        name_to_filename={strip_paths(x): x for x in filenames},
        metadata=[{'filename': x, 'year': int(x[2:6])} for x in basenames],
    )


@pytest.fixture
def source_info() -> cr.SourceInfo:
    return create_test_source_info(
        filenames=['/tmp/a_2001.txt', '/tmp/b_2001.txt', '/tmp/c_2002.txt', '/tmp/c_2002.csv']
    )


class TestSource(cr.ISource):
    def __init__(self, filenames: str):
        self.filenames = filenames
        self.within_context = False

    def namelist(self, *, pattern: str) -> List[str]:  # pylint: disable=unused-argument
        return [x for x in self.filenames if fnmatch(x, pattern)]

    def read(self, filename: str, as_binary: bool = False) -> Optional[Any]:  # pylint: disable=unused-argument
        return f'ipsum lupsum: {filename}'

    def exists(self, filename: str) -> bool:
        return filename in strip_paths(self.filenames)

    def __enter__(self):  # pylint: disable=unused-argument
        self.within_context = True

    def __exit__(self, _, value, traceback):  # pylint: disable=unused-argument
        self.within_context = False


def test_source_info(source_info: cr.SourceInfo):  # pylint: disable=redefined-outer-name
    assert source_info.to_stored_name('a_2000.txt') is None
    assert source_info.to_stored_name('a_2001.txt') == '/tmp/a_2001.txt'

    assert source_info.get_names(name_filter=[]) == []
    assert source_info.get_names(name_filter=['a_2000.txt']) == []
    assert source_info.get_names(name_filter=['a_2001.txt']) == ['a_2001.txt']

    assert source_info.get_metadata(name_filter=[]) == []
    assert source_info.get_metadata(name_filter=['a_2000.txt']) == []
    assert source_info.get_metadata(name_filter=['a_2001.txt']) == [{'filename': 'a_2001.txt', 'year': 2001}]


def test_source_get_info():
    source = TestSource(filenames=['/tmp/a_2001.txt', '/tmp/b_2001.txt', '/tmp/c_2002.txt', '/tmp/c_2002.csv'])

    info = source.get_info(opts=TextReaderOpts(filename_pattern='*.*'))
    assert info.names == ['a_2001.txt', 'b_2001.txt', 'c_2002.txt', 'c_2002.csv']

    info = source.get_info(opts=TextReaderOpts(filename_pattern='*.txt'))
    assert info.names == ['a_2001.txt', 'b_2001.txt', 'c_2002.txt']

    info = source.get_info(opts=TextReaderOpts(filename_pattern='*.csv'))
    assert info.names == ['c_2002.csv']

    info = source.get_info(opts=TextReaderOpts(filename_pattern='*.txt'))
    assert all(m['filename'] == info.names[i] for i, m in enumerate(info.metadata))
    assert all(info.name_to_filename[f] == os.path.join('/tmp', f) for f in info.names)


def test_sources():
    os.makedirs('./tests/output', exist_ok=True)

    def test_source(source):
        assert set(source.namelist()) == set(EXPECTED_TEXT_FILES + ['README.md'])
        assert set(source.namelist(pattern='*.txt')) == set(EXPECTED_TEXT_FILES)
        assert set(source.namelist(pattern='*.md')) == set(['README.md'])
        assert source.exists('dikt_2019_03_test.txt')

        with source as s:
            text = s.read('dikt_2019_02_test.txt')
        assert text.startswith('På väg i det långa mörkret.')
        assert text.endswith('ur med tidens fångna insekt.')

    source = cs.ZipSource(source_path=TEST_CORPUS_FILENAME)
    test_source(source)

    zip_utils.unpack(path=TEST_CORPUS_FILENAME, target_folder='./tests/output', create_sub_folder=True)
    source = cs.FolderSource(source_path=os.path.join(OUTPUT_FOLDER, strip_path_and_extension(TEST_CORPUS_FILENAME)))
    test_source(source)

    items = list(streamify_zip_source(path=TEST_CORPUS_FILENAME))
    source = cs.InMemorySource(items=items)
    test_source(source)

    items = list(streamify_zip_source(path=TEST_CORPUS_FILENAME, filename_pattern='*.*'))
    df = pd.DataFrame(data=items, columns=['filename', 'txt'])
    source = cs.PandasSource(data=df, text_column='txt', filename_column='filename')
    test_source(source)


def test_corpus_reader():
    filenames = ['/tmp/a_2001.txt', '/tmp/b_2001.txt', '/tmp/c_2002.txt', '/tmp/c_2002.csv']
    source = TestSource(filenames=filenames)

    store = cr.CorpusReader(source, TextReaderOpts(filename_pattern='*.txt'))
    assert store.filenames == ['a_2001.txt', 'b_2001.txt', 'c_2002.txt']

    store = cr.CorpusReader(source, TextReaderOpts(filename_pattern='*.csv'))
    assert store.filenames == ['c_2002.csv']

    store = cr.CorpusReader(source, TextReaderOpts(filename_pattern='*.txt')).apply_filter(['b_2001.txt'])
    assert store.filenames == ['b_2001.txt']

    store = cr.CorpusReader(source, TextReaderOpts(filename_pattern='*.txt')).apply_filter([])
    assert store.filenames == []

    store = cr.CorpusReader(source, TextReaderOpts(filename_pattern='*.txt')).apply_filter(
        ['a_2001.txt', 'b_2001.txt', 'c_2002.txt', 'c_2002.csv']
    )
    assert store.filenames == ['a_2001.txt', 'b_2001.txt', 'c_2002.txt']

    store = cr.CorpusReader(source, TextReaderOpts(filename_pattern='*.txt', filename_fields="year:_:1"))
    result = [x for x in store]
    assert [filename for filename, _ in result] == ['a_2001.txt', 'b_2001.txt', 'c_2002.txt']
    assert [x for x in store] == [x for x in store] == result
    assert store[1] == store['b_2001.txt'] == ('b_2001.txt', 'ipsum lupsum: /tmp/b_2001.txt') == store.item(1)
    assert len(store) == 3
    assert store.metadata[2] == {'filename': 'c_2002.txt', 'year': 2002}
    assert len([x for x in store.items]) == len(store)
    assert store.document_index is not None
    assert store.document_index.filename.tolist() == ['a_2001.txt', 'b_2001.txt', 'c_2002.txt']
    assert store.document_index.year.tolist() == [2001, 2001, 2002]
    assert store.document_index.columns.tolist() == ['filename', 'year', 'document_id', 'document_name']
    assert store.document_index.index.tolist() == store.document_index.document_name.tolist()
