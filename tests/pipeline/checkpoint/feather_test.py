import os

import pandas as pd

from penelope.pipeline.checkpoint import feather
from penelope.utility import strip_paths

FEATHER_DOCUMENT_INDEX_NAME = 'document_index.feathering'

TEST_FOLDER: str = 'tests/test_data/tranströmer/tranströmer_id_tagged_frames'


def test_get_document_filenames():
    filenames: list[str] = feather.get_document_filenames(folder=TEST_FOLDER)
    assert len(filenames) == 5

    assert all(x.startswith('tran_') for x in strip_paths(filenames))


def test_get_document_index_filename():
    filename = feather.get_document_index_filename(TEST_FOLDER)
    assert filename is not None
    assert strip_paths(filename).startswith('document_index.feather')


def test_document_index_exists():
    assert feather.document_index_exists(TEST_FOLDER) is True


def test_read_document_index():
    di: pd.DataFrame = feather.read_document_index(TEST_FOLDER)
    assert di is not None
    assert len(di) == 5


def test_write_document_index():
    di: pd.DataFrame = feather.read_document_index(TEST_FOLDER)
    assert di is not None

    feather.write_document_index('tests/output', di)

    di_stored: pd.DataFrame = feather.read_document_index('tests/output')
    assert di_stored is not None
    assert len(di_stored) == 5
    assert set(di.columns) == set(di_stored.columns)

    os.remove('tests/output/document_index.feathering')


def test_is_complete():
    assert feather.is_complete(TEST_FOLDER) is True
