from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Sequence, Union

from penelope import utility

from .corpus_mixins import PartitionMixIn
from .document_index import DocumentIndex, metadata_to_document_index, update_document_index_token_counts
from .interfaces import ITokenizedCorpus
from .readers.interfaces import ICorpusReader
from .tokens_transformer import TokensTransformer, TokensTransformOpts
from .utils import generate_token2id

logger = utility.getLogger("__penelope__")


class TokenizedCorpus(ITokenizedCorpus, PartitionMixIn):
    def __init__(self, reader: ICorpusReader, *, tokens_transform_opts: TokensTransformOpts = None):
        """[summary]

        Parameters
        ----------
        reader : ICorpusReader
            Corpus reader
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
        self._document_index: DocumentIndex = metadata_to_document_index(reader.metadata)
        self.transformer = TokensTransformer(tokens_transform_opts=(tokens_transform_opts or TokensTransformOpts()))
        self.iterator = None
        self._token2id = None

    def _create_document_tokens_stream(self):
        token_counts = []
        for filename, tokens in self.reader:
            raw_tokens = [x for x in tokens]
            cooked_tokens = [x for x in self.transformer.transform(raw_tokens)]
            token_counts.append((filename, len(raw_tokens), len(cooked_tokens)))
            yield filename, cooked_tokens
        self._document_index = update_document_index_token_counts(self._document_index, token_counts)

    def _create_iterator(self):
        return self._create_document_tokens_stream()

    @property
    def terms(self) -> Iterator[Iterator[str]]:
        return ReiterableTerms(self)

    @property
    def document_index(self) -> DocumentIndex:
        return self._document_index

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self.reader.metadata

    @property
    def filenames(self) -> List[str]:
        return self.reader.filenames

    @property
    def document_names(self) -> List[str]:
        return [utility.strip_path_and_extension(x) for x in self.reader.filenames]

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
        return len(self.document_index)

    @property
    def token2id(self):
        if self._token2id is None:
            self._token2id = generate_token2id(self.terms, len(self))
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
