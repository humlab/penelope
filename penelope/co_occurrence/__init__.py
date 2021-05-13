# type: ignore


from . import partition_by_document, partition_by_key
from .convert import to_co_occurrence_matrix, to_trends_data
from .hal_or_glove import GloveVectorizer, HyperspaceAnalogueToLanguageVectorizer, compute_hal_or_glove_co_occurrences
from .interface import ComputeResult, ContextOpts, CoOccurrenceError
from .persistence import (
    CO_OCCURRENCE_FILENAME_PATTERN,
    CO_OCCURRENCE_FILENAME_POSTFIX,
    Bundle,
    create_options_bundle,
    filename_to_folder_and_tag,
    folder_and_tag_to_filename,
    load_bundle,
    load_co_occurrences,
    load_options,
    store_bundle,
    store_co_occurrences,
    tag_to_filename,
)
from .windows_corpus import WindowsCorpus, WindowsStream
from .windows_utility import corpus_to_windows, tokens_to_windows
