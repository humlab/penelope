from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Sequence, Union

import pandas as pd
from penelope import utility
from tqdm import tqdm

from .corpus_mixins import PartitionMixIn, UpdateTokenCountsMixIn
from .interfaces import ITokenizedCorpus
from .readers.interfaces import ICorpusReader
from .tokens_transformer import TokensTransformer, TokensTransformOpts

logger = utility.getLogger("__penelope__")


class TokenizedCorpus(ITokenizedCorpus, PartitionMixIn, UpdateTokenCountsMixIn):
    def __init__(self, reader: ICorpusReader, *, tokens_transform_opts: TokensTransformOpts = None):
        """[summary]

        Parameters
        ----------
        reader : ICorpusReader
            Corpus tokenizer/reader
        tokens_transform_opts : TokensTransformOpts
            Passed to TokensTransformer and can be:
                only_alphabetic: bool = False,
                only_any_alphanumeric: bool = False,
                to_lower: bool = False,
                to_upper: bool = False,
                min_len: int = None,
                max_len: int = None,
                remove_accents: bool = False,
                remove_stopwords: bool = False,
                stopwords: Any = None,
                extra_stopwords: List[str] = None,
                language: str = "swedish",
                keep_numerals: bool = True,
                keep_symbols: bool = True,
        Raises
        ------
        TypeError
            Readers does not conform to ICorpusReader
        """
        if not hasattr(reader, 'metadata'):
            raise TypeError(f"Corpus reader {type(reader)} has no `metadata` property")

        if not hasattr(reader, 'filenames'):
            raise TypeError(f"Corpus reader {type(reader)} has no `filenames` property")

        self.reader: ICorpusReader = reader
        self._documents: pd.DataFrame = pd.DataFrame(reader.metadata)

        if 'document_id' not in self._documents:
            self._documents['document_id'] = list(self._documents.index)

        self.transformer = TokensTransformer(tokens_transform_opts=(tokens_transform_opts or TokensTransformOpts()))
        self.iterator = None
        self._token2id = None

    def _create_document_tokens_stream(self):

        doc_token_counts = []

        for filename, tokens in self.reader:

            tokens = [x for x in tokens]
            n_raw_tokens = len(tokens)

            tokens = [x for x in self.transformer.transform(tokens)]
            n_tokens = len(tokens)

            doc_token_counts.append((filename, n_raw_tokens, n_tokens))

            yield filename, tokens

        self._documents = self.update_token_counts(doc_token_counts)

    def _create_iterator(self):
        return self._create_document_tokens_stream()

    @property
    def terms(self) -> Iterator[Iterator[str]]:
        return ReiterableTerms(self)

    @property
    def documents(self) -> pd.DataFrame:
        return self._documents

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self.reader.metadata

    @property
    def filenames(self) -> List[str]:
        return self.reader.filenames

    def apply_filter(self, filename_filter: Union[str, Callable, Sequence]):
        if not hasattr(self.reader, 'apply_filter'):
            raise TypeError("apply_filter only valid for ICorpusReader")
        self.reader.apply_filter(filename_filter)

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

    def __len__(self):
        return len(self.documents)

    def _generate_token2id(self):
        token2id = defaultdict()
        token2id.default_factory = token2id.__len__
        tokens_iter = tqdm(self.terms, desc="Vocab", total=len(self))
        for tokens in tokens_iter:
            for token in tokens:
                _ = token2id[token]
            tokens_iter.set_description(f"Vocab #{len(token2id)}")
        return dict(token2id)

    @property
    def token2id(self):
        if self._token2id is None:
            self._token2id = self._generate_token2id()
        return self._token2id

    @property
    def id2token(self):
        return {v: k for k, v in self.token2id.items()}

    @property
    def id_terms(self) -> Iterator[Iterator[int]]:
        """Yields document as a token ID stream """
        t2id = self.token2id
        for tokens in self.terms:
            yield (t2id[token] for token in tokens)


class ReiterableTerms:
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
