from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .tokens_transformer import DEFAULT_PROCESS_OPTS, TokensTransformer


class TokenizedCorpus:
    def __init__(self, reader: Any, **kwargs):

        if not hasattr(reader, 'metadata'):
            raise TypeError(f"Corpus reader {type(reader)} has no `metadata` property")

        if not hasattr(reader, 'filenames'):
            raise TypeError(f"Corpus reader {type(reader)} has no `filenames` property")

        self.reader = reader
        self.documents = pd.DataFrame([d.__dict__ for d in reader.metadata])

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

        self.documents['n_raw_tokens'] = n_raw_tokens
        self.documents['n_tokens'] = n_tokens

    def _create_iterator(self):
        return self._create_document_tokens_stream()

    @property
    def terms(self):
        return ReIterableTerms(self)

    @property
    def metadata(self):
        return self.reader.metadata

    @property
    def filenames(self):
        return self.reader.filenames

    @property
    def n_raw_tokens(self) -> Dict:
        # FIXME: #1 Implement tokenized_corpus.n_raw_tokens
        # raise NotImplementedError("n_raw_tokens not implemented")
        return None

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
