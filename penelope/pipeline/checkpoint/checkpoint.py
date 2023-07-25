import json
import os
import zipfile
from dataclasses import asdict
from enum import StrEnum
from functools import cached_property
from io import StringIO
from os.path import dirname
from typing import Any, Callable, Iterable, Iterator, Optional

from loguru import logger

from penelope.corpus import DocumentIndex, DocumentIndexHelper, TextReaderOpts, Token2Id, load_document_index, serialize
from penelope.utility import filenames_satisfied_by, zip_utils

from ..interfaces import ContentType, DocumentPayload, PipelineError
from .load import PayloadLoader, load_payloads_multiprocess, load_payloads_singleprocess


class CheckpointNames(StrEnum):
    OPTIONS = "options.json"
    DOCUMENT_INDEX = "document_index.csv"
    DICTIONARY = "dictionary.csv"


class CorpusCheckpoint:
    """Container/Proxy for pipeline checkpoint data"""

    def __init__(
        self,
        *,
        source_name: Any = None,
        filenames: list[str] = None,
        document_index: DocumentIndex = None,
        token2id: Token2Id = None,
        serialize_opts: serialize.SerializeOpts = None,
        reader_opts: TextReaderOpts = None,
    ):
        self.source_name: Any = source_name
        self.document_index: DocumentIndex = document_index
        self.token2id: Token2Id = token2id
        self.serialize_opts: serialize.SerializeOpts = serialize_opts
        self.filenames: list[str] = filenames
        self.reader_opts: TextReaderOpts = reader_opts
        self.content_type: ContentType = serialize_opts.content_type

        self.document_index: DocumentIndex = (
            self.document_index
            if self.document_index is not None
            else DocumentIndexHelper.from_filenames2(self.filenames, self.reader_opts)
        )

        self._sync_filenames()
        self._filter_documents()

        self.create_stream: Callable[[], Iterable[DocumentPayload]] = self._payload_stream_abstract_factory()

    def _payload_stream_abstract_factory(self) -> PayloadLoader:
        load_payload_stream = (
            load_payloads_multiprocess if self.serialize_opts.deserialize_processes else load_payloads_singleprocess
        )
        return lambda: load_payload_stream(
            zip_or_filename=self.source_name, opts=self.serialize_opts, filenames=self.filenames
        )

    def _filter_documents(self) -> None:
        """Filter documents and document index based on criterias in reader_opts"""

        if self.reader_opts is not None:
            self.filenames = filenames_satisfied_by(
                self.filenames,
                filename_filter=self.reader_opts.filename_filter,
                filename_pattern=self.reader_opts.filename_pattern,
            )

        if self.document_index is not None:
            self.document_index = self.document_index[self.document_index.filename.isin(self.filenames)]

    def _sync_filenames(self, verbose: bool = True) -> None:
        """Syncs sort order for archive filenames and filenames in document index"""

        if self.document_index is None:
            return

        if self.filenames == self.document_index.filename.to_list():
            return

        if set(self.filenames) != set(self.document_index.filename.to_list()):
            raise ValueError(f"{self.source_name} filenames in archive and document index differs")

        if 'document_id' in self.document_index:
            self.document_index.sort_values(by=['document_id'], inplace=True)

        if not verbose:
            logger.warning(f"{self.source_name} filename sort order mismatch (using document index sort order)")

        self.filenames = self.document_index.filename.to_list()


def store_archive(
    *,
    opts: serialize.SerializeOpts,
    target_filename: str,
    document_index: DocumentIndex,
    payload_stream: Iterator[DocumentPayload],
    token2id: Token2Id = None,
    compresslevel: int = 8,
) -> Iterable[DocumentPayload]:
    """Store payload stream as a compressed ZIP archive"""
    target_folder: bool = dirname(target_filename)
    serializer: serialize.IContentSerializer = serialize.SerializerRegistry.create(opts)

    os.makedirs(target_folder, exist_ok=True)

    with zipfile.ZipFile(
        target_filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel
    ) as zf:
        zf.writestr(CheckpointNames.OPTIONS, json.dumps(asdict(opts)).encode('utf8'))

        for payload in payload_stream:
            zf.writestr(payload.filename, data=serializer.serialize(content=payload.content, options=opts))
            yield payload

        if document_index is not None:
            document_index_name = opts.document_index_name or CheckpointNames.DOCUMENT_INDEX
            document_index_sep = opts.document_index_sep or "\t"
            zf.writestr(document_index_name, data=document_index.to_csv(sep=document_index_sep, header=True))

        if token2id is not None:
            zf.writestr(CheckpointNames.DICTIONARY, data=json.dumps(token2id.data))


def load_archive(
    source_name: str, opts: serialize.SerializeOpts = None, reader_opts: TextReaderOpts = None
) -> CorpusCheckpoint:
    """Load a TAGGED FRAME corpus stored in a ZIP FILE with CSV-files and optionally a document index"""

    with CheckpointZipFile(source_name, mode="r", opts=opts) as cp:
        return CorpusCheckpoint(
            source_name=source_name,
            filenames=cp.document_filenames,
            document_index=cp.document_index,
            token2id=cp.token2id,
            serialize_opts=cp.opts or opts,
            reader_opts=reader_opts,
        )


class CheckpointZipFile(zipfile.ZipFile):
    def __init__(self, file: Any, mode: str, opts: serialize.SerializeOpts = None) -> None:
        super().__init__(file, mode=mode, compresslevel=zipfile.ZIP_DEFLATED)

        self.opts: serialize.SerializeOpts = opts or self.load_opts()

        if self.opts is None:
            raise PipelineError(
                f"Checkpoint options not supplied and file {CheckpointNames.OPTIONS} not found in archive."
            )

    def load_opts(self) -> Optional[serialize.SerializeOpts]:
        """Returns checkpoint options stored in archive, or None if not found in archive"""

        if CheckpointNames.OPTIONS not in self.namelist():
            return None

        data: dict = zip_utils.read_json(zip_or_filename=self, filename=CheckpointNames.OPTIONS)
        opts = serialize.SerializeOpts.create(data)

        return opts

    @cached_property
    def document_index_name(self) -> str:
        return self.opts.document_index_name or CheckpointNames.DOCUMENT_INDEX

    @cached_property
    def document_index(self) -> Optional[DocumentIndex]:
        """Returns the document index stored in archive, or None if not exists"""
        if self.document_index_name not in self.namelist():
            return None

        return load_document_index(
            StringIO(
                zip_utils.read_file_content(zip_or_filename=self, filename=self.document_index_name, as_binary=False)
            ),
            sep=self.opts.document_index_sep,
        )

    @cached_property
    def token2id(self) -> Optional[Token2Id]:
        """Returns dictionary stored in archive, or None if not found in archive"""

        if CheckpointNames.DICTIONARY not in self.namelist():
            return None

        return Token2Id(zip_utils.read_json(zip_or_filename=self, filename=CheckpointNames.DICTIONARY))

    @cached_property
    def document_filenames(self) -> list[str]:
        return [
            f
            for f in self.namelist()
            if f not in [self.document_index_name, CheckpointNames.OPTIONS, CheckpointNames.DICTIONARY]
        ]
