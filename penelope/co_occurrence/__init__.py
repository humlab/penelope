# type: ignore
from .co_occurrence import corpus_co_occurrence, corpus_to_windows, tokens_to_windows
from .convert import (
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
    to_co_occurrence_matrix,
    to_dataframe,
    to_trends_data,
    to_vectorized_corpus,
)
from .hal_or_glove import GloveVectorizer, HyperspaceAnalogueToLanguageVectorizer, compute_hal_or_glove_co_occurrences
from .interface import ContextOpts, CoOccurrenceError
from .partitioned import ComputeResult, partitioned_corpus_co_occurrence
from .windows_corpus import WindowsCorpus, WindowsStream
