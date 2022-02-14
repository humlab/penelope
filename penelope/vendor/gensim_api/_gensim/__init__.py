# type: ignore

from .utils import (
    Dictionary,
    Sparse2Corpus,
    corpus2csc,
    from_id2token_to_dictionary,
    from_stream_of_tokens_to_dictionary,
    from_stream_of_tokens_to_sparse2corpus,
)

try:
    from .ext_text_corpus import ExtTextCorpus, SimpleExtTextCorpus
except (ImportError, NameError):
    ...
