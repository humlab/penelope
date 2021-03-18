from typing import Callable, Iterable, Iterator, List, Union

from penelope.utility import getLogger

from ...document_index import DocumentIndex, metadata_to_document_index
from ..interfaces import ICorpusReader, TextReaderOpts
from .interfaces import ContentFilter, ISource, StoreItemPair
from .sources import SourceInfo
from .transformer import TextTransformer

logger = getLogger("penelope")

# pylint: disable=too-many-arguments,too-many-instance-attributes


class CorpusReader(ICorpusReader):
    """Gateway/proxy for a file store that can be filtered, streamed and indexed.
    Applying a filter creates a view of a subset of the source files that affects all retrieval methods.
    Note that the source must be able to retrive data by index (name or integer)
    The data can be indexed even when filters are applied:
        filenames[i]    the i:th filename in the store
        metadata[i]     collected metadata fot i:th file
        self[i]         content of i:th file, or content of named file
    The i:th element in filenames, metadata and self
    """

    def __init__(
        self,
        source: ISource,
        reader_opts: TextReaderOpts = None,
        transformer: TextTransformer = None,
        preprocess: Callable[[str], str] = None,
        tokenizer: Callable[[str], Iterator[str]] = None,
    ):

        self.source: ISource = source
        self.reader_opts: TextReaderOpts = reader_opts.copy() if reader_opts is not None else TextReaderOpts()
        self.transformer: TextTransformer = transformer
        self.iterator = None
        self.preprocess: Callable[[str], str] = preprocess
        self.tokenizer: Callable[[str], Iterator[str]] = tokenizer
        self.source_info: SourceInfo = source.get_info(self.reader_opts)

    @property
    def filenames(self) -> List[str]:
        return self.source_info.get_names(
            name_filter=self.reader_opts.filename_filter,
            name_pattern=self.reader_opts.filename_pattern,
        )

    @property
    def metadata(self):
        return self.source_info.get_metadata(
            name_filter=self.reader_opts.filename_filter,
            name_pattern=self.reader_opts.filename_pattern,
        )

    def apply_filter(self, filename_filter: ContentFilter) -> "CorpusReader":
        self.reader_opts.filename_filter = filename_filter
        return self

    @property
    def items(self) -> Iterable[StoreItemPair]:
        with self.source:
            return (self.item(name) for name in self.filenames)

    @property
    def document_index(self) -> DocumentIndex:
        index: DocumentIndex = metadata_to_document_index(
            self.metadata,
            document_id_field=self.reader_opts.index_field,
        )
        return index

    def item(self, name: str) -> StoreItemPair:

        name = self.filenames[name] if isinstance(name, int) else name

        data = self.source.read(
            self.source_info.to_stored_name(name),
            self.reader_opts.as_binary,
        )

        if self.preprocess:
            data = self.preprocess(data)

        if self.transformer and isinstance(data, str):
            data = self.transformer.transform(data)

        if self.tokenizer is not None:
            data = self.tokenizer(data)

        return (name, data)

    def __getitem__(self, item: Union[str, int]) -> StoreItemPair:
        return self.item(item)

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        self.iterator = self.items
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise
        except TypeError as ex:
            if self.iterator is None:
                raise TypeError("tips: next() called without prior call to iter()") from ex
            raise
