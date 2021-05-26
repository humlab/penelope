import pytest

from penelope.corpus.dtm import VectorizedCorpus
from penelope.co_occurrence import to_filename, Bundle
from penelope.notebook.co_occurrence.tabular_gui import CoOccurrenceTable, get_prepared_corpus

# pylint: disable=protected-access

@pytest.fixture
def bundle():
    folder, tag = './tests/test_data/VENUS', 'VENUS'
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle


def test_get_prepared_corpus(bundle):

    corpus: VectorizedCorpus = get_prepared_corpus(
        corpus=bundle.corpus,
        period_specifier="year",
        tf_idf=False,
        token_filter="",
        global_threshold=1,
    )

    assert corpus is not None

def test_table_gui_create(bundle):


    gui: CoOccurrenceTable = CoOccurrenceTable(bundle=bundle)

    assert gui is not None
    assert gui.bundle is bundle
    assert gui.co_occurrences is None

    gui.alert("🤢")
    gui.info("😊")

@pytest.mark.parametrize("category", ["year", "lustrum", "decade"])
def test_table_gui_to_corpus(bundle, category):

    gui: CoOccurrenceTable = CoOccurrenceTable(bundle=bundle)

    gui.stop_observe()
    gui.pivot = category
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert "category" in corpus.document_index.columns

    gui.stop_observe()
    gui.tf_idf = True
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert "category" in corpus.document_index.columns

    gui.stop_observe()
    gui.token_filter = "apa"
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert corpus.data.shape == (5, 0)

    gui.stop_observe()
    gui.token_filter = "educational/*"
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert all(x.startswith("educational") for x in corpus.token2id)

    gui.stop_observe()
    gui.tf_idf = False
    gui.token_filter = ""
    gui.global_threshold = 100
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert all(corpus.term_frequencies >= 100)

    gui.save()

@pytest.mark.parametrize("category", ["year", "lustrum", "decade"])
def test_table_gui_to_co_occurrences(bundle, category):

    gui: CoOccurrenceTable = CoOccurrenceTable(bundle=bundle)

    gui.stop_observe()
    gui.pivot = category
    gui.tf_idf = False
    gui.token_filter = "educational/*"
    gui.global_threshold = 50
    gui.concepts = set(["general"])
    gui.largest = 10
    gui.start_observe()

    gui._update_corpus()

    assert len(gui.co_occurrences) > 0
    assert all(gui.co_occurrences.token.str.startswith("educational"))
    assert all(gui.co_occurrences.w1 == "educational")

