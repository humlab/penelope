import os
import shutil
import uuid

import pandas as pd

import penelope.co_occurrence as co_occurrence
from penelope.corpus import VectorizedCorpus
from tests.fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS, very_simple_corpus
from tests.utils import OUTPUT_FOLDER

from . import utils as test_utils

jj = os.path.join


def test_filename_to_folder_and_tag():
    filename = f'./tests/test_data/dummy/dummy{co_occurrence.FILENAME_POSTFIX}'

    folder, tag = co_occurrence.to_folder_and_tag(filename)

    assert folder == './tests/test_data/dummy'
    assert tag == 'dummy'


def test_folder_and_tag_to_filename():
    expected_filename: str = f'./tests/test_data/dummy/dummy{co_occurrence.FILENAME_POSTFIX}'

    folder, tag = './tests/test_data/dummy', 'dummy'

    filename: str = co_occurrence.to_filename(folder=folder, tag=tag)

    assert filename == expected_filename


def test_load_co_occurrences():
    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    co_occurrences: pd.DataFrame = co_occurrence.load_co_occurrences(filename)

    assert co_occurrences is not None
    assert 16399 == len(co_occurrences)
    assert 125197 == co_occurrences.value.sum()


def test_load_options():
    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename: str = co_occurrence.to_filename(folder=folder, tag=tag)

    options = co_occurrence.load_options(filename)

    assert options is not None


def test_store_co_occurrences():
    filename: str = f'VENUS{co_occurrence.FILENAME_POSTFIX}'
    target_folder: str = f'./tests/output/{uuid.uuid4()}'

    source_filename: str = jj('./tests/test_data/VENUS', filename)
    target_filename: str = jj(target_folder, filename)

    co_occurrences = co_occurrence.load_co_occurrences(source_filename)

    os.makedirs(target_folder)

    co_occurrence.store_co_occurrences(filename=target_filename, co_occurrences=co_occurrences)

    assert os.path.isfile(target_filename)

    co_occurrences = co_occurrence.load_co_occurrences(target_filename)
    assert co_occurrences is not None

    shutil.rmtree(target_folder, ignore_errors=True)


def test_load_and_store_bundle():
    filename = co_occurrence.to_filename(folder='./tests/test_data/VENUS', tag='VENUS')
    target_folder: str = f'./tests/output/{uuid.uuid4()}'

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

    assert bundle is not None
    assert isinstance(bundle.corpus, VectorizedCorpus)
    assert isinstance(bundle.co_occurrences, pd.DataFrame)
    assert isinstance(bundle.compute_options, dict)
    assert bundle.folder == './tests/test_data/VENUS'
    assert bundle.tag == 'VENUS'

    os.makedirs(target_folder)

    expected_filename = co_occurrence.to_filename(folder=target_folder, tag='MARS')

    bundle.store(folder=target_folder, tag='MARS')

    assert os.path.isfile(expected_filename)

    shutil.rmtree(target_folder, ignore_errors=True)


def test_compute_and_store_bundle():
    tag: str = f'{uuid.uuid4()}'

    target_folder: str = jj(OUTPUT_FOLDER, tag)
    target_filename: str = co_occurrence.to_filename(folder=target_folder, tag=tag)

    os.makedirs(target_folder, exist_ok=True)

    simple_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        concept={'g'}, ignore_concept=False, context_width=2
    )
    bundle: co_occurrence.Bundle = test_utils.create_simple_bundle_by_pipeline(
        data=simple_corpus,
        context_opts=context_opts,
        folder=target_folder,
        tag=tag,
    )

    bundle.store()

    assert os.path.isfile(target_filename)

    shutil.rmtree(target_folder, ignore_errors=True)
