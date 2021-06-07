import os

import pandas as pd
import penelope.co_occurrence as co_occurrence
import pytest
from penelope.corpus import VectorizedCorpus
from tests.fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS, very_simple_corpus
from tests.utils import OUTPUT_FOLDER

from . import utils as test_utils

jj = os.path.join


def test_filename_to_folder_and_tag():

    filename = f'./tests/test_data/VENUS/VENUS{co_occurrence.FILENAME_POSTFIX}'

    folder, tag = co_occurrence.to_folder_and_tag(filename)

    assert folder == './tests/test_data/VENUS'
    assert tag == 'VENUS'


def test_folder_and_tag_to_filename():

    expected_filename: str = f'./tests/test_data/VENUS/VENUS{co_occurrence.FILENAME_POSTFIX}'

    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename: str = co_occurrence.to_filename(folder=folder, tag=tag)

    assert filename == expected_filename


def test_load_co_occurrences():

    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    co_occurrences: pd.DataFrame = co_occurrence.load_co_occurrences(filename)

    assert co_occurrences is not None
    assert 16070 == len(co_occurrences)
    assert 123142 == co_occurrences.value.sum()


def test_load_options():

    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename: str = co_occurrence.to_filename(folder=folder, tag=tag)

    options = co_occurrence.load_options(filename)

    assert options is not None


@pytest.mark.skip(reason="not implemented")
def test_create_options_bundle():
    pass


def test_store_co_occurrences():

    filename = f'VENUS{co_occurrence.FILENAME_POSTFIX}'

    source_filename = jj('./tests/test_data/VENUS', filename)
    target_filename = jj('./tests/output', filename)

    co_occurrences = co_occurrence.load_co_occurrences(source_filename)

    co_occurrence.store_co_occurrences(filename=target_filename, co_occurrences=co_occurrences)

    assert os.path.isfile(target_filename)

    co_occurrences = co_occurrence.load_co_occurrences(target_filename)
    assert co_occurrences is not None

    os.remove(target_filename)


def test_load_and_store_bundle():

    filename = co_occurrence.to_filename(folder='./tests/test_data/VENUS', tag='VENUS')

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

    assert bundle is not None
    assert isinstance(bundle.corpus, VectorizedCorpus)
    assert isinstance(bundle.co_occurrences, pd.DataFrame)
    assert isinstance(bundle.compute_options, dict)
    assert bundle.folder == './tests/test_data/VENUS'
    assert bundle.tag == 'VENUS'

    os.makedirs('./tests/output/MARS', exist_ok=True)

    expected_filename = co_occurrence.to_filename(folder='./tests/output/MARS', tag='MARS')

    bundle.store(folder='./tests/output/MARS', tag='MARS')

    assert os.path.isfile(expected_filename)


def test_compute_and_store_bundle():

    tag: str = "JUPYTER"
    folder: str = jj(OUTPUT_FOLDER, tag)
    filename: str = co_occurrence.to_filename(folder=folder, tag=tag)

    os.makedirs(folder, exist_ok=True)

    simple_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        concept={'g'}, ignore_concept=False, context_width=2
    )
    bundle: co_occurrence.Bundle = test_utils.create_co_occurrence_bundle(
        corpus=simple_corpus,
        context_opts=context_opts,
        folder=folder,
        tag=tag,
    )

    bundle.store()

    assert os.path.isfile(filename)

    os.remove(filename)
