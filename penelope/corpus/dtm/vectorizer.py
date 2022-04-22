from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Literal, Mapping, Tuple, Union

import more_itertools
import numpy as np
import pandas as pd

from penelope.utility import PropsMixIn, list_to_unique_list_with_preserved_order, strip_path_and_extension

from ..document_index import DocumentIndex
from ..tokenized_corpus import TokenizedCorpus
from .corpus import VectorizedCorpus

try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    ...


DocumentTermsStream = Iterable[Tuple[str, Iterable[str]]]


@dataclass
class VectorizeOpts(PropsMixIn["VectorizeOpts"]):
    already_tokenized: bool = True
    lowercase: bool = False
    stop_words: str = None
    max_df: float = 1.0
    min_df: int = 1
    min_tf: int = 1
    max_tokens: int = None


def _no_tokenize(tokens):
    return tokens


def _no_tokenize_lowercase(tokens):
    return [t.lower() for t in tokens]


def isiterable(x: Any) -> bool:
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False


def check_tokens_stream(head: Any) -> None:

    if not isinstance(head, tuple):
        raise ValueError(f"expected str ✕ text/tokens, found {type(head)}")

    if len(head) != 2:
        raise ValueError(f"expected str ✕ text/tokens, found {' ✕ '.join(type(x) for x in head)}]")

    if not isinstance(head[0], str):
        raise ValueError(f"expected str (document name) as first element in tuple {type(head[1])}")

    if isinstance(head[1], str):
        raise ValueError(
            f"expected Iterable as second element when already_tokenized is set to true, found {type(head[1])}"
        )

    if not isiterable(head[1]):
        raise ValueError(f"expected Iterable[str] when already_tokenized is True but found {type(head[1])}")


class CorpusVectorizer:
    def __init__(self):
        self.vectorizer = None
        self.vectorizer_opts = {}

    def fit_transform_(
        self,
        corpus: Union[TokenizedCorpus, DocumentTermsStream],
        *,
        vocabulary: Mapping[str, int] = None,
        document_index: Union[Callable[[], DocumentIndex], DocumentIndex] = None,
        vectorize_opts: VectorizeOpts,
    ) -> VectorizedCorpus:
        """Same as `fit_transform` but with a parameter object"""
        return self.fit_transform(corpus, vocabulary=vocabulary, document_index=document_index, **vectorize_opts.props)

    def fit_transform(
        self,
        corpus: Union[TokenizedCorpus, DocumentTermsStream],
        *,
        already_tokenized: bool = True,
        vocabulary: Mapping[str, int] = None,
        max_tokens: int = None,
        document_index: Union[Callable[[], DocumentIndex], DocumentIndex] = None,
        lowercase: bool = False,
        stop_words: Union[Literal['english'], List[str]] = None,
        max_df: float = 1.0,
        min_df: int = 1,
        min_tf: int = 1,
        dtype: Any = np.int32,
        tokenizer: Callable[[str], Iterable[str]] = None,
        token_pattern=r"(?u)\b\w+\b",
    ) -> VectorizedCorpus:
        """Returns a `VectorizedCorpus` (document-term-matrix, bag-of-word) by applying sklearn's `CountVecorizer` on `corpus`
        If `already_tokenized` is True then the input stream is expected to be tokenized.
        Input stream sort order __MUST__ be the same as document_index sort order.
        Passed `document_index` can be a callable that returns a DocumentIndex. This is necessary
        for instance when document index isn't avaliable until pipeline is exhausted.

        Args:
            corpus (Union[TokenizedCorpus, DocumentTermsStream]): Stream of text or stream of tokens
            already_tokenized (bool, optional): Specifies if stream is tokens. Defaults to True.
            vocabulary (Mapping[str, int], optional): Predefined vocabulary. Defaults to None.
            document_index (Union[Callable[[], DocumentIndex], DocumentIndex], optional): If callable, then resolved after the stream has been exhausted. Defaults to None.
            lowercase (bool, optional): Let vectorizer lowercase text. Defaults to False.
            stop_words (str, optional): Let vectorizer remove stopwords. Defaults to None.
            max_df (float, optional): Max document frequency (see CountVecorizer). Defaults to 1.0.
            min_df (int, optional): Min document frequency (see CountVecorizer). Defaults to 1.
            min_tf (int, optional): Min term frequency. Defaults to None.
            max_tokens (int, optional): Restrict to top max tokens (see `max_features` in CountVectorizer). Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            VectorizedCorpus: [description]

        Yields:
            Iterator[VectorizedCorpus]: [description]
        """
        tokenizer: Callable[[str], Iterable[str]] = None
        if vocabulary is None:
            if hasattr(corpus, 'vocabulary'):
                vocabulary = corpus.vocabulary
            elif hasattr(corpus, 'token2id'):
                vocabulary = corpus.token2id

        if already_tokenized:

            heads, corpus = more_itertools.spy(corpus, n=1)

            if heads:
                check_tokens_stream(heads[0])

            if lowercase:
                tokenizer = _no_tokenize_lowercase
                lowercase = False
            else:
                tokenizer = _no_tokenize

        vectorizer_opts: dict = dict(
            tokenizer=tokenizer,
            lowercase=lowercase,
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            vocabulary=vocabulary,
            max_features=max_tokens,
            token_pattern=token_pattern,
            dtype=dtype,
        )

        seen_document_names: List[str] = []

        def terms_stream():
            for name, terms in corpus:
                seen_document_names.append(name)
                yield terms

        self.vectorizer = CountVectorizer(**vectorizer_opts)
        self.vectorizer_opts = vectorizer_opts

        bag_term_matrix = self.vectorizer.fit_transform(terms_stream())
        token2id: dict = self.vectorizer.vocabulary_

        document_index_: DocumentIndex = resolve_document_index(corpus, document_index, seen_document_names)

        dtm_corpus: VectorizedCorpus = VectorizedCorpus(
            bag_term_matrix,
            token2id=token2id,
            document_index=document_index_,
        )

        if min_tf and min_tf > 1:
            dtm_corpus = dtm_corpus.slice_by_tf(min_tf)

        return dtm_corpus


def resolve_document_index(
    source: Union[TokenizedCorpus, DocumentTermsStream],
    document_index: Union[Callable[[], DocumentIndex], DocumentIndex],
    seen_document_names: List[str],
) -> DocumentIndex:

    if document_index is None:
        for attr in ['documents', 'document_index']:
            if hasattr(source, attr) and getattr(source, attr) is not None:
                document_index = getattr(source, attr)

    _document_index = _set_strictly_increasing_index_by_seen_documents(
        document_index() if callable(document_index) else document_index,
        seen_document_names,
    )
    return _document_index


def _set_strictly_increasing_index_by_seen_documents(
    document_index: Union[Callable[[], DocumentIndex], DocumentIndex],
    seen_document_names: List[str],
) -> DocumentIndex:

    if document_index is None:

        # logger.warning("vectorizer: no corpus document index supplied: generating from seen filenames")
        seen_document_index: DocumentIndex = pd.DataFrame(
            {
                'filename': seen_document_names,
                'document_name': [strip_path_and_extension(x) for x in seen_document_names],
            }
        )
        seen_document_index['document_id'] = seen_document_index.index

        return seen_document_index

    # strip extension and remove duplicates (should only occur if chunked data) - we MUST keep document sequence
    seen_document_names: List[str] = [
        strip_path_and_extension(x) for x in list_to_unique_list_with_preserved_order(seen_document_names)
    ]

    # create {filename: sequence_id} map of the seen documents, ordered as they were seen/processed
    _recode_map = {x: i for i, x in enumerate(seen_document_names)}

    # filter out documents that wasn't processed
    document_index: DocumentIndex = document_index[document_index.document_name.isin(seen_document_names)]

    # recode document_id to sequence_id
    document_index['document_id'] = document_index['document_name'].apply(lambda x: _recode_map[x])

    # set 'document_id' as new index, and make sure it is sorted strictly increasing
    document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

    return document_index


# def _get_stream_length(
#     corpus: Union[TokenizedCorpus, DocumentTermsStream],
#     document_index: DocumentIndex,
# ) -> int:
#     if hasattr(corpus, '__len__'):
#         return len(corpus)
#     if hasattr(corpus, 'documents') and corpus.document_index is not None:
#         return len(corpus.document_index)
#     if hasattr(corpus, 'document_index') and corpus.document_index is not None:
#         return len(corpus.document_index)
#     if document_index is not None:
#         return len(document_index)
#     return None
