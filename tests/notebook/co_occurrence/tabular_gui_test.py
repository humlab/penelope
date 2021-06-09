import pytest
from penelope.co_occurrence import Bundle, to_filename
from penelope.common.keyness import KeynessMetric
from penelope.corpus import VectorizedCorpus
from penelope.notebook.co_occurrence.tabular_gui import TabularCoOccurrenceGUI

# pylint: disable=protected-access, redefined-outer-name


@pytest.fixture
def bundle():
    folder, tag = './tests/test_data/VENUS', 'VENUS'
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

    assert all(corpus.term_frequencies >= 100)

    gui.save()


@pytest.mark.parametrize("time_period", ["year", "lustrum", "decade"])
def test_table_gui_to_co_occurrences_filters_out_tokens(bundle, time_period):

    gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle)

    gui.stop_observe()
    gui.pivot = time_period
    gui.keyness = KeynessMetric.TF
    gui.token_filter = "educational/*"
    gui.global_threshold = 50
    gui.concepts = set(["general"])
    gui.largest = 10
    gui.start_observe()

    gui._update_corpus()

    co_occurrences = gui.to_filtered_co_occurrences()

    assert len(co_occurrences) > 0
    assert all(co_occurrences.token.str.startswith("educational"))
    assert all(co_occurrences.w1 == "educational")


@pytest.mark.parametrize(
    "folder,tag,keyness",
    [
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.PPMI),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.HAL_cwr),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.LLR),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.LLR_Dunning),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.DICE),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.TF_IDF),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.TF_normalized),
    ],
)
def test_table_gui_debug_setup(folder: str, tag: str, keyness: KeynessMetric):

    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)

    assert bundle is not None

    gui: TabularCoOccurrenceGUI = TabularCoOccurrenceGUI(bundle=bundle).setup()

    gui.stop_observe()
    gui.pivot = "year"
    gui.keyness = keyness
    gui.token_filter = ""
    gui.global_threshold = 1
    gui.concepts = set()
    gui.largest = 10
    gui.start_observe()

    gui._update_corpus()


@pytest.mark.parametrize(
    "folder,tag,keyness",
    [
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.PPMI),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.HAL_cwr),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.LLR),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.LLR_Dunning),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.DICE),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.TF_IDF),
        ('./tests/test_data/VENUS', 'VENUS', KeynessMetric.TF_normalized),
    ],
)
def test_get_prepared_corpus(folder: str, tag: str, keyness: KeynessMetric):
    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)

    corpus: VectorizedCorpus = bundle.to_keyness_corpus(
        period_pivot="year",
        keyness=keyness,
        global_threshold=1,
        pivot_column_name='time_period',
        normalize=False,
        fill_gaps=False,
    )

    assert corpus is not None
