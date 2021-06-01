import pytest
from penelope.co_occurrence import Bundle, to_filename
from penelope.corpus.dtm import VectorizedCorpus
from penelope.notebook.co_occurrence.tabular_gui import TabularCoOccurrenceGUI, KeynessMetric, get_prepared_corpus

# pylint: disable=protected-access, redefined-outer-name


@pytest.fixture
def bundle():
    folder, tag = './tests/test_data/VENUS', 'VENUS'
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle


def test_get_prepared_corpus(bundle):

    corpus: VectorizedCorpus = get_prepared_corpus(
        bundle=bundle,
        corpus=bundle.corpus,
        period_pivot="year",
        keyness=KeynessMetric.TF,
        token_filter="",
        global_threshold=1,
        pivot_column_name='time_period',
    )

    assert corpus is not None


def test_table_gui_create(bundle):

    gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle)

    assert gui is not None
    assert gui.bundle is bundle
    assert gui.co_occurrences is None

    gui.alert("🤢")
    gui.info("😊")


@pytest.mark.parametrize("time_period", ["year", "lustrum", "decade"])
def test_table_gui_to_corpus(bundle, time_period):

    gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle)

    gui.stop_observe()
    gui.pivot = time_period
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert "time_period" in corpus.document_index.columns

    gui.stop_observe()
    gui.tf_idf = True
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert "time_period" in corpus.document_index.columns

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


@pytest.mark.parametrize("time_period", ["year", "lustrum", "decade"])
def test_table_gui_to_co_occurrences(bundle, time_period):

    gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle)

    gui.stop_observe()
    gui.pivot = time_period
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
