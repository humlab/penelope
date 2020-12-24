import pandas as pd
import penelope.co_occurrence as co_occurrence
import penelope.notebook.co_occurrence as co_occurrences_gui
from penelope.corpus import VectorizedCorpus

DATA_FOLDER = './tests/test_data'


def test_create_load_co_occurrences_gui():
    def load_callback(_: str):
        pass

    gui = co_occurrences_gui.create_load_gui(data_folder=DATA_FOLDER)

    gui = gui.setup(filename_pattern=co_occurrence.CO_OCCURRENCE_FILENAME_PATTERN, load_callback=load_callback)
    assert gui is not None

    layout = gui.layout()

    assert layout is not None


def test_load_co_occurrence_bundle():

    filename = './tests/test_data/VENUS/VENUS_co-occurrence.csv.zip'

    bundle = co_occurrence.load_bundle(filename)

    assert bundle is not None
    assert isinstance(bundle.corpus, VectorizedCorpus)
    assert isinstance(bundle.co_occurrences, pd.DataFrame)
    assert isinstance(bundle.compute_options, dict)
    assert bundle.corpus_folder == './tests/test_data/VENUS'
    assert bundle.corpus_tag == 'VENUS'
