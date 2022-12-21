import json
import os
import zipfile
from dataclasses import asdict
from io import StringIO
from os.path import dirname
from typing import Any, Callable, Iterable, Iterator, List, Optional

from loguru import logger

from penelope.corpus import DocumentIndex, DocumentIndexHelper, TextReaderOpts, Token2Id, load_document_index
from penelope.utility import filenames_satisfied_by, zip_utils

from ..interfaces import ContentType, DocumentPayload, PipelineError
from .interface import (
    CHECKPOINT_OPTS_FILENAME,
    DICTIONARY_FILENAME,
    DOCUMENT_INDEX_FILENAME,
    CheckpointOpts,
    IContentSerializer,
)
from .load import PayloadLoader, load_payloads_multiprocess, load_payloads_singleprocess
from .serialize import create_serializer


class CheckpointData:
    """Container/Proxy for pipeline checkpoint data"""

    def __init__(
        self,
        *,
        source_name: Any = None,
        filenames: List[str] = None,
        document_index: DocumentIndex = None,
        token2id: Token2Id = None,
        checkpoint_opts: CheckpointOpts = None,
        payload_loader_override: PayloadLoader = None,
        reader_opts: TextReaderOpts = None,
    ):
        self.source_name: Any = source_name
        self.document_index: DocumentIndex = document_index
        self.token2id: Token2Id = token2id
        self.checkpoint_opts: CheckpointOpts = checkpoint_opts
        self.filenames: List[str] = filenames
        self.reader_opts: TextReaderOpts = reader_opts
        self.content_type: ContentType = checkpoint_opts.content_type

        self.document_index: DocumentIndex = (
            self.document_index
            if self.document_index is not None
            else DocumentIndexHelper.from_filenames2(self.filenames, self.reader_opts)
        )

        self._sync_filenames()
        self._filter_documents()

        self.create_stream: Callable[[], Iterable[DocumentPayload]] = self._payload_stream_abstract_factory(
            payload_loader_override
        )

    def _payload_stream_abstract_factory(self, payload_loader_override: PayloadLoader) -> PayloadLoader:
        load_payload_stream = payload_loader_override or (
            load_payloads_multiprocess if self.checkpoint_opts.deserialize_processes else load_payloads_singleprocess
        )
        return lambda: load_payload_stream(
            zip_or_filename=self.source_name, checkpoint_opts=self.checkpoint_opts, filenames=self.filenames
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
    checkpoint_opts: CheckpointOpts,
    target_filename: str,
    document_index: DocumentIndex,
    payload_stream: Iterator[DocumentPayload],
    token2id: Token2Id = None,
    compresslevel: int = 8,
) -> Iterable[DocumentPayload]:
    """Store payload stream as a compressed ZIP archive"""
    target_folder: bool = dirname(target_filename)
    serializer: IContentSerializer = create_serializer(checkpoint_opts)

    os.makedirs(target_folder, exist_ok=True)

    with zipfile.ZipFile(
        target_filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel
    ) as zf:

        zf.writestr(CHECKPOINT_OPTS_FILENAME, json.dumps(asdict(checkpoint_opts)).encode('utf8'))

        for payload in payload_stream:
            zf.writestr(payload.filename, data=serializer.serialize(content=payload.content, options=checkpoint_opts))
            yield payload

        if document_index is not None:
            document_index_name = checkpoint_opts.document_index_name or DOCUMENT_INDEX_FILENAME
            document_index_sep = checkpoint_opts.document_index_sep or "\t"
            zf.writestr(document_index_name, data=document_index.to_csv(sep=document_index_sep, header=True))

        if token2id is not None:
            zf.writestr(DICTIONARY_FILENAME, data=json.dumps(token2id.data))


def load_archive(
    source_name: str,
    checkpoint_opts: CheckpointOpts = None,
    reader_opts: TextReaderOpts = None,
    payload_loader: PayloadLoader = None,
) -> CheckpointData:
    """Load a TAGGED FRAME checkpoint stored in a ZIP FILE with CSV-filed and optionally a document index

    Args:
        source_name (str): [description]
        checkpoint_opts (CheckpointOpts, optional): deserialize opts. Defaults to None.
        reader_opts (TextReaderOpts, optional): Settings for creating a document index or filtering files. Defaults to None.

    Raises:
        PipelineError: Something is wrong

    Returns:
        CheckpointData: Checkpoint data contianer
    """

    with _CheckpointZipFile(source_name, mode="r", checkpoint_opts=checkpoint_opts) as zf:

        return CheckpointData(
            source_name=source_name,
            filenames=zf.document_filenames,
            document_index=zf.document_index,
            token2id=zf.token2id,
            checkpoint_opts=zf.checkpoint_opts or checkpoint_opts,
            payload_loader_override=payload_loader,
            reader_opts=reader_opts,
        )


class _CheckpointZipFile(zipfile.ZipFile):
    def __init__(
        self,
        file: Any,
        mode: str,
        checkpoint_opts: CheckpointOpts = None,
        document_index_name=DOCUMENT_INDEX_FILENAME,
    ) -> None:

        super().__init__(file, mode=mode, compresslevel=zipfile.ZIP_DEFLATED)

        self.checkpoint_opts: CheckpointOpts = checkpoint_opts or self._checkpoint_opts()

        if self.checkpoint_opts is None:
            raise PipelineError(
                f"Checkpoint options not supplied and file {CHECKPOINT_OPTS_FILENAME} not found in archive."
            )

        self.document_index_name: str = document_index_name or self.checkpoint_opts.document_index_name
        self.document_index: DocumentIndex = self._document_index()
        self.token2id: Token2Id = self._token2id()

    def _checkpoint_opts(self) -> Optional[CheckpointOpts]:
        """Returns checkpoint options stored in archive, or None if not found in archive"""

        if CHECKPOINT_OPTS_FILENAME not in self.namelist():
            return None

        _opts_dict: dict = zip_utils.read_json(zip_or_filename=self, filename=CHECKPOINT_OPTS_FILENAME)
        _opts = CheckpointOpts.load(_opts_dict)

        return _opts

    def _document_index(self) -> Optional[DocumentIndex]:
        """Returns the document index stored in archive, or None if not exists"""
        if self.document_index_name not in self.namelist():
            return None

        return load_document_index(
            StringIO(
                zip_utils.read_file_content(zip_or_filename=self, filename=self.document_index_name, as_binary=False)
            ),
            sep=self.checkpoint_opts.document_index_sep,
        )

    def _token2id(self) -> Optional[Token2Id]:
        """Returns dictionary stored in archive, or None if not found in archive"""

        if DICTIONARY_FILENAME not in self.namelist():
            return None

        return Token2Id(zip_utils.read_json(zip_or_filename=self, filename=DICTIONARY_FILENAME))

    @property
    def document_filenames(self) -> List[str]:
        return [f for f in self.namelist() if f not in [self.document_index_name, CHECKPOINT_OPTS_FILENAME]]
