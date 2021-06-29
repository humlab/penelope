import pytest
from penelope.co_occurrence import Bundle, to_filename
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.corpus import VectorizedCorpus
from penelope.notebook.co_occurrence.tabular_gui import TabularCoOccurrenceGUI

# pylint: disable=protected-access, redefined-outer-name


@pytest.fixture
def bundle():
    folder, tag = './tests/test_data/SSI', 'SSI'
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle


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
    gui.keyness = KeynessMetric.TF
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert "time_period" in corpus.document_index.columns

    gui.stop_observe()
    gui.keyness = KeynessMetric.TF
    gui.global_threshold = 100
    gui.start_observe()

    corpus: VectorizedCorpus = gui.to_corpus()

    assert ((corpus.term_frequency >= 100) | (corpus.term_frequency == 0)).all()

    gui.save()


@pytest.mark.parametrize("time_period", ["year", "lustrum", "decade"])
def test_table_gui_to_co_occurrences_filters_out_tokens(bundle, time_period):

    gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle)

    gui.stop_observe()
    gui.pivot = time_period
    gui.keyness_source = KeynessMetricSource.Full
    gui.keyness = KeynessMetric.TF
    gui.token_filter = "educational/*"
    gui.global_threshold = 50
    gui.concepts = set(["general"])
    gui.largest = 10
    gui.start_observe()

    gui.update_corpus()
    gui.update_co_occurrences()

    co_occurrences = gui.to_filtered_co_occurrences()

    assert len(co_occurrences) > 0
    assert all(co_occurrences.token.str.startswith("educational"))
    assert all(co_occurrences.w1 == "educational")


@pytest.mark.long_running
@pytest.mark.parametrize(
    "tag,keyness",
    [
        ('SSI', KeynessMetric.PPMI),
        ('SSI', KeynessMetric.HAL_cwr),
        ('SSI', KeynessMetric.LLR),
        ('SSI', KeynessMetric.LLR_Dunning),
        ('SSI', KeynessMetric.DICE),
        ('SSI', KeynessMetric.TF_IDF),
        ('SSI', KeynessMetric.TF),
        ('SSI', KeynessMetric.TF_normalized),
    ],
)
def test_table_gui_debug_setup(tag: str, keyness: KeynessMetric):

    folder: str = f'./tests/test_data/{tag}'

    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)

    assert bundle is not None

    gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle)

    gui.stop_observe()
    gui.pivot = "year"
    gui.keyness_source = KeynessMetricSource.Full
    gui.keyness = keyness
    gui.token_filter = ""
    gui.global_threshold = 1
    gui.concepts = set()
    gui.largest = 10
    gui.start_observe()

    gui.update_corpus()
