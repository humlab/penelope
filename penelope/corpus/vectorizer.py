import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Tuple, Union

import pandas as pd
from penelope.corpus import metadata_to_document_index
from penelope.utility import PropsMixIn
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.std import tqdm

from .tokenized_corpus import TokenizedCorpus
from .vectorized_corpus import VectorizedCorpus

logger = logging.getLogger("corpus_vectorizer")


DocumentTermsStream = Iterable[Tuple[str, Iterable[str]]]


@dataclass
class VectorizeOpts(PropsMixIn):
    tokenizer: Callable = None
    lowercase: bool = False
    stop_words: str = None
    max_df: float = 1.0
    min_df: int = 1
    verbose: bool = True


def _no_tokenize(tokens):
    return tokens


class CorpusVectorizer:
    def __init__(self):
        self.vectorizer = None
        self.vectorizer_opts = {}

    def fit_transform_(
        self,
        corpus: Union[TokenizedCorpus, DocumentTermsStream],
        *,
        vocabulary: Mapping[str, int] = None,
        documents: pd.DataFrame = None,
        vectorize_opts: VectorizeOpts,
    ) -> VectorizedCorpus:
        """Same as `fit_transform` but with a parameter object """
        return self.fit_transform(corpus, vocabulary=vocabulary, documents=documents, **vectorize_opts.props)

    def fit_transform(
        self,
        corpus: Union[TokenizedCorpus, DocumentTermsStream],
        *,
        vocabulary: Mapping[str, int] = None,
        documents: pd.DataFrame = None,
        tokenizer: Callable = None,
        lowercase: bool = False,
        stop_words: str = None,
        max_df: float = 1.0,
        min_df: int = 1,
        verbose: bool = True,
    ) -> VectorizedCorpus:
        """Returns a `VectorizedCorpus` (document-term-matrix, bag-of-word) by applying sklearn's `CountVecorizer` on `corpus`

                Note:
        `
                  - Input stream is expected to be already tokenized if `tokenizer` is None
                  - Input stream sort order __MUST__ be the same as document_index sort order

                Parameters
                ----------
                corpus : tokenized_corpus.TokenizedCorpus
                    [description]
                max_df: Union[int,float], float in range [0.0, 1.0] or int, default=1.0
                    Frequent words filter. sklearn: "Ignore terms that have a document frequency strictly higher than the given threshold"
                min_df: Union[int,float], float in range [0.0, 1.0] or int, default=1
                    Rare words filter, sklearn: "ignore terms that have a document frequency strictly lower than the given threshold"
                Returns
                -------
                vectorized_corpus.VectorizedCorpus
                    [description]
        """

        if vocabulary is None:
            if hasattr(corpus, 'vocabulary'):
                vocabulary = corpus.vocabulary
            elif hasattr(corpus, 'token2id'):
                vocabulary = corpus.token2id

        if tokenizer is None:
            tokenizer = _no_tokenize
            if lowercase:
                tokenizer = lambda tokens: [t.lower() for t in tokens]
            lowercase = False

        vectorizer_opts = dict(
            tokenizer=tokenizer,
            lowercase=lowercase,
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            vocabulary=vocabulary,
        )

        seen_document_filenames = []

        def terms_stream():
            # document_terms_stream = (
            #     zip(corpus.documents.filename.to_list(), corpus.terms) if hasattr(corpus, 'terms') else corpus
            # )
            for filename, terms in corpus:  # document_terms_stream:
                seen_document_filenames.append(filename)
                yield terms

        terms = terms_stream()

        # if verbose:
        #     terms = tqdm(terms, total=_get_stream_length(corpus, documents))

        self.vectorizer = CountVectorizer(**vectorizer_opts)
        self.vectorizer_opts = vectorizer_opts

        bag_term_matrix = self.vectorizer.fit_transform(terms)
        token2id = self.vectorizer.vocabulary_

        v_document_index = _consolidate_document_index(corpus, documents, seen_document_filenames)

        # We need to recode document indexso so that dooument_id corresponds to DTM document row number
        v_document_index = _document_index_recode_id(v_document_index, seen_document_filenames)

        v_corpus = VectorizedCorpus(bag_term_matrix, token2id, v_document_index)

        return v_corpus


def _document_index_recode_id(document_index: pd.DataFrame, document_names: List[str]) -> pd.DataFrame:

    _recode_map = {x: i for i, x in enumerate(document_names)}
    document_index['document_id'] = document_index['filename'].apply(lambda x: _recode_map[x])
    return document_index.sort_values('document_id')


def _supplied_document_index(
    corpus: Union[TokenizedCorpus, DocumentTermsStream], documents_index: pd.DataFrame
) -> pd.DataFrame:

    if documents_index is not None:
        return documents_index

    for attr in ['documents', 'document_index']:
        if hasattr(corpus, attr):
            if getattr(corpus, attr) is not None:
                return getattr(corpus, attr)

    return None


def unique_list_with_preserved_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def _consolidate_document_index(
    corpus: Union[TokenizedCorpus, DocumentTermsStream],
    documents_index: pd.DataFrame,
    seen_document_filenames: List[str],
) -> pd.DataFrame:

    supplied_index: pd.DataFrame = _supplied_document_index(corpus, documents_index)

    if supplied_index is not None:
        if supplied_index.index.tolist() != seen_document_filenames:
            if supplied_index.index.tolist() != unique_list_with_preserved_order(seen_document_filenames):
                logger.warning('"bug-check: documents_index mismatch (supplied/seen differs)"')
                # raise ValueError("bug-check: documents_index mismatch (supplied/seen differs)")
        return supplied_index

    logger.warning("no corpus document index supplied: generating from seen filenames")

    seen_document_index: pd.DataFrame = metadata_to_document_index(
        [dict(filename=filename) for filename in seen_document_filenames]
    )

    return seen_document_index


def _get_stream_length(
    corpus: Union[TokenizedCorpus, DocumentTermsStream],
    document_index: pd.DataFrame,
) -> int:
    if hasattr(corpus, '__len__'):
        return len(corpus)
    if hasattr(corpus, 'documents') and corpus.documents is not None:
        return len(corpus.documents)
    if hasattr(corpus, 'document_index') and corpus.document_index is not None:
        return len(corpus.documents)
    if document_index is not None:
        return len(document_index)
    return None
