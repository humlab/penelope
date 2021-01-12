# type: ignore
from penelope.pipeline.config import CorpusType

from .compute_corpus import compute_co_occurrence as corpus_compute_co_occurrence
from .compute_pipeline import POS_CHECKPOINT_FILENAME_POSTFIX
from .compute_pipeline import compute_co_occurrence as pipeline_compute_co_occurrence
from .explore_co_occurrence_gui import ExploreGUI
from .load_co_occurrences_gui import LoadGUI, create_load_gui
from .to_co_occurrence_gui import ComputeGUI


def compute_pipeline_factory(corpus_type: CorpusType):

    if corpus_type == CorpusType.SpacyCSV:
        return pipeline_compute_co_occurrence

    if corpus_type == CorpusType.SparvCSV:
        return corpus_compute_co_occurrence

    raise ValueError(f"Unsupported CorpusType: {corpus_type}")
