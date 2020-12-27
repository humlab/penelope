import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple, Union

import more_itertools
import pandas as pd
from penelope.utility import PropsMixIn, list_to_unique_list_with_preserved_order, strip_path_and_extension
from sklearn.feature_extraction.text import CountVectorizer

from ..tokenized_corpus import TokenizedCorpus
from .vectorized_corpus import VectorizedCorpus

logger = logging.getLogger("corpus_vectorizer")


DocumentTermsStream = Iterable[Tuple[str, Iterable[str]]]


@dataclass
class VectorizeOpts(PropsMixIn):
    already_tokenized: bool = True
    lowercase: bool = False
    stop_words: str = None
    max_df: float = 1.0
    min_df: int = 1
    verbose: bool = True


def _no_tokenize(tokens):
    return tokens


def _no_tokenize_lowercase(tokens):
    return [t.lower() for t in tokens]


class CorpusVectorizer:
    def __init__(self):
        self.vectorizer = None
        self.vectorizer_opts = {}

    def fit_transform_(
        self,
        corpus: Union[TokenizedCorpus, DocumentTermsStream],
        *,
        vocabulary: Mapping[str, int] = None,
        document_index: pd.DataFrame = None,
        vectorize_opts: VectorizeOpts,
    ) -> VectorizedCorpus:
        """Same as `fit_transform` but with a parameter object """
        return self.fit_transform(corpus, vocabulary=vocabulary, document_index=document_index, **vectorize_opts.props)

    def fit_transform(
        self,
        corpus: Union[TokenizedCorpus, DocumentTermsStream],
        *,
        already_tokenized: bool = True,
        vocabulary: Mapping[str, int] = None,
        document_index: pd.DataFrame = None,
        lowercase: bool = False,
        stop_words: str = None,
        max_df: float = 1.0,
        min_df: int = 1,
        verbose: bool = True,  # pylint: disable=unused-argument
    ) -> VectorizedCorpus:
        """Returns a `VectorizedCorpus` (document-term-matrix, bag-of-word) by applying sklearn's `CountVecorizer` on `corpus`

                Note:
        `
                  - If `already_tokenized` is True then the input stream is expected to be tokenized
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
                dtm.VectorizedCorpus
                    [description]
        """
        if document_index is None:
            for attr in ['documents', 'document_index']:
                if hasattr(corpus, attr) and getattr(corpus, attr) is not None:
                    document_index = getattr(corpus, attr)

        if vocabulary is None:
            if hasattr(corpus, 'vocabulary'):
                vocabulary = corpus.vocabulary
            elif hasattr(corpus, 'token2id'):
                vocabulary = corpus.token2id

        if already_tokenized:
            head, corpus = more_itertools.spy(corpus)
            if len(head) > 0 and isinstance(head[0][1], str):
                raise ValueError("CorpusVectorizer expects List[str] when already_tokenized is True but found str")
            if lowercase:
                tokenizer = _no_tokenize_lowercase
                lowercase = False
            else:
                tokenizer = _no_tokenize
        else:
            tokenizer = None

        vectorizer_opts = dict(
            tokenizer=tokenizer,
            lowercase=lowercase,
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            vocabulary=vocabulary,
        )

        seen_document_names = []

        def terms_stream():
            for name, terms in corpus:  # document_terms_stream:
                seen_document_names.append(name)
                yield terms

        terms = terms_stream()

        # if verbose:
        #     terms = tqdm(terms, total=_get_stream_length(corpus, document_index))

        self.vectorizer = CountVectorizer(**vectorizer_opts)
        self.vectorizer_opts = vectorizer_opts

        bag_term_matrix = self.vectorizer.fit_transform(terms)
        token2id = self.vectorizer.vocabulary_

        v_document_index = _set_strictly_increasing_index_by_seen_documents(document_index, seen_document_names)

        v_corpus = VectorizedCorpus(bag_term_matrix, token2id, v_document_index)

        return v_corpus


def _set_strictly_increasing_index_by_seen_documents(
    document_index: pd.DataFrame, seen_document_names: List[str]
) -> pd.DataFrame:

    if document_index is None:

        # logger.warning("vectorizer: no corpus document index supplied: generating from seen filenames")
        seen_document_index: pd.DataFrame = pd.DataFrame(
            {
                'filename': seen_document_names,
                'document_name': [strip_path_and_extension(x) for x in seen_document_names],
            }
        )
        seen_document_index['document_id'] = seen_document_index.index

        return seen_document_index

    # strp extension and remove duplicates (should only occur if chunked data) - we MUST keep document sequence
    seen_document_names = [
        strip_path_and_extension(x) for x in list_to_unique_list_with_preserved_order(seen_document_names)
    ]

    # create {filename: sequence_id} map of the seen documents, ordered as they were seen/processed
    _recode_map = {x: i for i, x in enumerate(seen_document_names)}

    # filter out documents that wasn't processed
    document_index = document_index[document_index.document_name.isin(seen_document_names)]

    # recode document_id to sequence_id
    document_index['document_id'] = document_index['document_name'].apply(lambda x: _recode_map[x])

    # set 'document_id' as new index, and make sure it is sorted strictly increasing
    document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

    return document_index


def _get_stream_length(
    corpus: Union[TokenizedCorpus, DocumentTermsStream],
    document_index: pd.DataFrame,
) -> int:
    if hasattr(corpus, '__len__'):
        return len(corpus)
    if hasattr(corpus, 'documents') and corpus.document_index is not None:
        return len(corpus.document_index)
    if hasattr(corpus, 'document_index') and corpus.document_index is not None:
        return len(corpus.document_index)
    if document_index is not None:
        return len(document_index)
    return None
