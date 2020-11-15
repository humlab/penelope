import logging
from typing import Callable, Iterable, Mapping, Tuple, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.std import tqdm

from .tokenized_corpus import TokenizedCorpus
from .vectorized_corpus import VectorizedCorpus

logger = logging.getLogger("corpus_vectorizer")


# def _default_tokenizer(lowercase=True):
#     def _lowerccase_tokenize(tokens):
#         return [x.lower() for x in tokens]

#     def _no_tokenize(tokens):
#         return tokens

#     if lowercase:
#         return lambda tokens: [x.lower() for x in tokens]

#     return _lowerccase_tokenize if lowercase else _no_tokenize


def _no_tokenize(tokens):
    return tokens


DocumentTermsStream = Iterable[Tuple[str, Iterable[str]]]


class CorpusVectorizer:
    def __init__(self):
        self.vectorizer = None
        self.vectorizer_opts = {}

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
        """Returns a vectorized corpus from of `corpus`

        Note:
          -  Input texts are already tokenized, so tokenizer is an identity function

        Parameters
        ----------
        corpus : tokenized_corpus.TokenizedCorpus
            [description]

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

        if tokenizer is None:  # Iterator[Tuple[str,Iterator[str]]]
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

        if hasattr(corpus, 'terms'):
            terms = corpus.terms
        else:
            terms = (x[1] for x in corpus)

        self.vectorizer = CountVectorizer(**vectorizer_opts)
        self.vectorizer_opts = vectorizer_opts

        if verbose:
            total = None
            if hasattr(corpus, '__len__'):
                total = len(corpus)
            elif hasattr(corpus, 'documents'):
                total = len(corpus.documents) if corpus.documents is not None else None

            terms = tqdm(terms, total=total, desc="Vectorizing: ")

        bag_term_matrix = self.vectorizer.fit_transform(terms)
        token2id = self.vectorizer.vocabulary_

        documents = documents if documents is not None else (corpus.documents if hasattr(corpus, 'documents') else None)

        if documents is None:
            logger.warning("corpus has no `documents` property (generating a dummy index")
            documents = pd.DataFrame(
                data=[{'index': i, 'filename': f'file_{i}.txt'} for i in range(0, bag_term_matrix.shape[0])]
            ).set_index('index')
            documents['document_id'] = documents.index

        # ignored_words = self.vectorizer.stop_words_

        v_corpus = VectorizedCorpus(bag_term_matrix, token2id, documents)

        return v_corpus


# FXIME: Deprecate this function (user _vectorize_corpus workflow instead)
