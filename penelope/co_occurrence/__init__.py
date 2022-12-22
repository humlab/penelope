# type: ignore
from .bundle import Bundle
from .convert import term_term_matrix_to_co_occurrences, to_co_occurrence_matrix, truncate_by_global_threshold
from .hal_or_glove import GloveVectorizer, HyperspaceAnalogueToLanguageVectorizer, compute_hal_or_glove_co_occurrences
from .interface import ContextOpts, CoOccurrenceError, Token, ZeroComputeError
from .persistence import (
    DICTIONARY_POSTFIX,
    DOCUMENT_INDEX_POSTFIX,
    FILENAME_PATTERN,
    FILENAME_POSTFIX,
    WindowCountDTM,
    create_options_bundle,
    load_co_occurrences,
    load_document_index,
    load_options,
    store_co_occurrences,
    store_document_index,
    to_filename,
    to_folder_and_tag,
)
from .prepare import CoOccurrenceHelper
from .vectorize import VectorizedTTM, VectorizeType, windows_to_ttm
from .windows import WindowsCorpus, generate_windows
