# type: ignore
# pylint: disable=unused-import,unused-argument
# flake8: noqa

from __future__ import annotations

from typing import Any, AnyStr, Callable, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from penelope.utility import DummyClass, streamify_any_source


class TextCorpus(DummyClass):
    ...


class MmCorpus(DummyClass):
    ...


class Dictionary(DummyClass):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.token2id: dict = None

    @staticmethod
    def from_corpus(corpus, id2word=None):  # pylint: disable=unused-argument
        raise ModuleNotFoundError()


class MMCorpus(DummyClass):
    @staticmethod
    def from_corpus(corpus, id2word=None):  # pylint: disable=unused-argument
        raise ModuleNotFoundError()


class Sparse2Corpus:
    # Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
    # Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
    # Modified code;
    def __init__(self, sparse, documents_columns=True):
        self.sparse = sparse.tocsc() if documents_columns else sparse.tocsr().T

    def __iter__(self):
        for indprev, indnow in zip(self.sparse.indptr, self.sparse.indptr[1:]):
            yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def __len__(self):
        return self.sparse.shape[1]

    def __getitem__(self, document_index):
        indprev = self.sparse.indptr[document_index]
        indnow = self.sparse.indptr[document_index + 1]
        return list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))


def corpus2csc(
    corpus,
    num_terms=None,
    dtype=np.float64,
    num_docs=None,
    num_nnz=None,
    printprogress=0,  # pylint: disable=unused-argument
):
    # Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
    # Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
    # (code removed)
    return None


try:

    from gensim.corpora.dictionary import Dictionary
    from gensim.corpora.mmcorpus import MmCorpus
    from gensim.corpora.textcorpus import TextCorpus
    from gensim.matutils import Sparse2Corpus, corpus2csc

    has_gensim: bool = True
except (ImportError, NameError):
    has_gensim: bool = False

# pylint: disable=abstract-method
class ExtTextCorpus(TextCorpus):
    def __init__(
        self,
        stream: Iterable[Tuple[str, AnyStr]],
        dictionary: dict = None,
        metadata=False,
        character_filters=None,
        tokenizer: Callable[[str], List[str]] = None,
        token_filters=None,
        bigram_transform=False,
    ):
        self.stream: Iterable[Tuple[str, AnyStr]] = stream
        self.filenames: List[str] = None
        self.document_index: pd.DataFrame = None
        self.length: int = None

        # if 'filenames' in content_iterator.__dict__:
        #    self.filenames = content_iterator.filenames
        #    self.document_names = self._compile_documents()
        #    self.length = len(self.filenames)

        token_filters = self.default_token_filters() + (token_filters or [])

        # if bigram_transform is True:
        #    train_corpus = GenericTextCorpus(content_iterator, token_filters=[ x.lower() for x in tokens ])
        #    phrases = gensim.models.phrases.Phrases(train_corpus)
        #    bigram = gensim.models.phrases.Phraser(phrases)
        #    token_filters.append(
        #        lambda tokens: bigram[tokens]
        #    )

        super().__init__(
            input=True,
            dictionary=dictionary,
            metadata=metadata,
            character_filters=character_filters,
            tokenizer=tokenizer,
            token_filters=token_filters,
        )

    def default_token_filters(self):
        return [
            (lambda tokens: [x.lower() for x in tokens]),
            (lambda tokens: [x for x in tokens if any(map(lambda x: x.isalpha(), x))]),
        ]

    def getstream(self):
        document_infos = []
        for filename, content in self.stream:
            yield content
            document_infos.append({'document_name': filename})

        self.length = len(document_infos)
        self.document_index: pd.DataFrame = pd.DataFrame(document_infos)
        self.filenames: List[str] = list(self.document_index.document_name.values)

    def get_texts(self):
        for document in self.getstream():
            yield self.preprocess_text(document)

    def preprocess_text(self, text) -> List[str]:

        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens: List[str] = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        return tokens

    def __get_document_info(self, filename):
        return {
            'document_name': filename,
        }


class SimpleExtTextCorpus(ExtTextCorpus):
    """Reads content in stream and returns tokenized text. No other processing."""

    def __init__(self, source: Any, lowercase: bool = False, filename_filter=None):

        self.reader: Iterable[Tuple[str, AnyStr]] = streamify_any_source(source, filename_filter=filename_filter)
        self.filenames: List[str] = self.reader.filenames
        self.lowercase: bool = lowercase

        super().__init__(self.reader)

    def default_token_filters(self) -> List[Callable]:

        token_filters = [
            (lambda tokens: [x.strip('_') for x in tokens]),
        ]

        if self.lowercase:
            token_filters = token_filters + [(lambda tokens: [x.lower() for x in tokens])]

        return token_filters

    def preprocess_text(self, text: str) -> List[str]:
        return self.tokenizer(text)


def _id2token2token2id(id2token: Mapping[int, str]) -> dict:
    if id2token is None:
        return None
    if hasattr(id2token, 'token2id'):
        return id2token.token2id
    token2id: dict = {v: k for k, v in id2token.items()}
    return token2id


GensimBowCorpus = Iterable[Iterable[Tuple[int, float]]]


def from_stream_of_tokens_to_sparse2corpus(source: Any, vocabulary: Dictionary | dict) -> Sparse2Corpus:

    if not hasattr(vocabulary, 'doc2bow'):
        vocabulary: Dictionary = _from_token2id_to_dictionary(vocabulary)

    bow_corpus: GensimBowCorpus = [vocabulary.doc2bow(tokens) for _, tokens in source]
    csc_matrix: sp.csc_matrix = corpus2csc(
        bow_corpus,
        num_terms=len(vocabulary),
        num_docs=len(bow_corpus),
        num_nnz=sum(map(len, bow_corpus)),
    )
    corpus: Sparse2Corpus = Sparse2Corpus(csc_matrix, documents_columns=True)
    return corpus


def from_stream_of_tokens_to_dictionary(source: Any, id2token: dict) -> Dictionary:
    """Creates a Dictionary from source using existing `id2token` mapping.
    Useful if cfs/dfs are needed, otherwise just use the existing mapping."""
    vocabulary: Dictionary = Dictionary()
    if id2token is not None:
        vocabulary.token2id = _id2token2token2id(id2token)
    vocabulary.add_documents(tokens for _, tokens in source)
    return vocabulary


def _from_token2id_to_dictionary(token2id: Mapping[str, int]) -> Dictionary:

    if isinstance(token2id, Dictionary):
        return token2id

    dictionary: Dictionary = Dictionary()
    dictionary.token2id = token2id

    return dictionary


def from_id2token_to_dictionary(id2token: dict) -> Dictionary:
    """Creates a `Dictionary` from a id2token dict."""
    return _from_token2id_to_dictionary(_id2token2token2id(id2token))
