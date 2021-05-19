import os

import pandas as pd
import penelope.co_occurrence as co_occurrence
import penelope.corpus.dtm as dtm

jj = os.path.join


def test_filename_to_folder_and_tag():

    filename = f'./tests/test_data/VENUS/VENUS{co_occurrence.FILENAME_POSTFIX}'

    folder, tag = co_occurrence.to_folder_and_tag(filename)

    assert folder == './tests/test_data/VENUS'
    assert tag == 'VENUS'


def test_folder_and_tag_to_filename():

    expected_filename = f'./tests/test_data/VENUS/VENUS{co_occurrence.FILENAME_POSTFIX}'

    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    assert filename == expected_filename


def test_load_co_occurrences():

    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    co_occurrences = co_occurrence.load_co_occurrences(filename)

    assert co_occurrences is not None
    assert 16070 == len(co_occurrences)
    assert 123142 == co_occurrences.value.sum()


def test_store_co_occurrences():

    filename = f'VENUS{co_occurrence.FILENAME_POSTFIX}'

    source_filename = jj('./tests/test_data/VENUS', filename)
    target_filename = jj('./tests/output', filename)

    co_occurrences = co_occurrence.load_co_occurrences(source_filename)

    co_occurrence.store_co_occurrences(target_filename, co_occurrences)

    assert os.path.isfile(target_filename)

    co_occurrences = co_occurrence.load_co_occurrences(target_filename)
    assert co_occurrences is not None

    os.remove(target_filename)


def test_load_and_store_bundle():

    filename = co_occurrence.to_filename(folder='./tests/test_data/VENUS', tag='VENUS')

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

    assert bundle is not None
    assert isinstance(bundle.corpus, dtm.VectorizedCorpus)
    assert isinstance(bundle.co_occurrences, pd.DataFrame)
    assert isinstance(bundle.compute_options, dict)
    assert bundle.folder == './tests/test_data/VENUS'
    assert bundle.tag == 'VENUS'

    os.makedirs('./tests/output/MARS', exist_ok=True)

    expected_filename = co_occurrence.to_filename(folder='./tests/output/MARS', tag='MARS')

    bundle.store(folder='./tests/output/MARS', tag='MARS')

    assert bundle.co_occurrence_filename == expected_filename
    assert os.path.isfile(bundle.co_occurrence_filename)
