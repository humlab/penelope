from .compute_hal_or_glove import compute
# from .compute_partitioned import (compute_for_column_group, load_text_windows,
#                                   partitoned_cooccurrence)
from .term_term_matrix import to_coocurrence_matrix, to_dataframe
from .vectorizer_glove import GloveVectorizer
from .vectorizer_hal import HyperspaceAnalogueToLanguageVectorizer
