from __future__ import annotations

from typing import Any, Dict, Iterator, List

import pandas as pd

from penelope.interfaces import ICorpusReader

from .interfaces import ITokenizedCorpus
from .tokens_transformer import DEFAULT_PROCESS_OPTS, TokensTransformer


class TokenizedCorpus(ITokenizedCorpus):
    def __init__(self, reader: ICorpusReader, **kwargs):

        if not hasattr(reader, 'metadata'):
            raise TypeError(f"Corpus reader {type(reader)} has no `metadata` property")

        if not hasattr(reader, 'filenames'):
            raise TypeError(f"Corpus reader {type(reader)} has no `filenames` property")

        self.reader = reader
        self._documents = pd.DataFrame(reader.metadata)

        if 'document_id' not in self._documents:
            self._documents['document_id'] = list(self._documents.index)

        opts = DEFAULT_PROCESS_OPTS
        opts = {**opts, **{k: v for k, v in kwargs.items() if k in opts}}

        self.transformer = TokensTransformer(**opts)
        self.iterator = None

    def _create_document_tokens_stream(self):

        n_raw_tokens = []
        n_tokens = []
        for dokument_name, tokens in self.reader:

            tokens = [x for x in tokens]

            n_raw_tokens.append(len(tokens))

            tokens = self.transformer.transform(tokens)

            tokens = [x for x in tokens]

            n_tokens.append(len(tokens))

            yield dokument_name, tokens

        self._documents['n_raw_tokens'] = n_raw_tokens
        self._documents['n_tokens'] = n_tokens

    def _create_iterator(self):
        return self._create_document_tokens_stream()

    @property
    def terms(self) -> Iterator[Iterator[str]]:
        return ReIterableTerms(self)

    @property
    def documents(self) -> pd.DataFrame:
        return self._documents

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self.reader.metadata

    @property
    def filenames(self) -> List[str]:
        return self.reader.filenames

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self._create_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise


class ReIterableTerms:
    def __init__(self, corpus):

        self.corpus = corpus
        self.iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = (tokens for _, tokens in self.corpus)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise
