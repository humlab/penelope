# type: ignore


from . import partition_by_document  # , partition_by_key
from .convert import to_co_occurrence_matrix, to_trends_data
from .hal_or_glove import GloveVectorizer, HyperspaceAnalogueToLanguageVectorizer, compute_hal_or_glove_co_occurrences
from .interface import ContextOpts, CoOccurrenceComputeResult, CoOccurrenceError, Token, ZeroComputeError
from .persistence import (
    DICTIONARY_POSTFIX,
    DOCUMENT_INDEX_POSTFIX,
    FILENAME_PATTERN,
    FILENAME_POSTFIX,
    Bundle,
    create_options_bundle,
    load_co_occurrences,
    load_document_index,
    load_options,
    store_co_occurrences,
    store_document_index,
    to_filename,
    to_folder_and_tag,
)
from .windows import WindowsCorpus, WindowsStream, corpus_to_windows, tokens_to_windows
