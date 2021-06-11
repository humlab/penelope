import json
import zipfile
from dataclasses import asdict
from io import StringIO
from os.path import basename
from typing import Any, Callable, Iterable, Iterator, List, Optional

from penelope.corpus import DocumentIndex, DocumentIndexHelper, TextReaderOpts, Token2Id, load_document_index
from penelope.utility import assert_that_path_exists, filenames_satisfied_by, getLogger, path_of, zip_utils

from ..interfaces import DocumentPayload, PipelineError
from .interface import (
    CHECKPOINT_OPTS_FILENAME,
    DICTIONARY_FILENAME,
    DOCUMENT_INDEX_FILENAME,
    CheckpointData,
    CheckpointOpts,
    IContentSerializer,
)
from .serialize import create_serializer, deserialized_payload_stream, parallel_deserialized_payload_stream

logger = getLogger("penelope")


def store_checkpoint(
    *,
    checkpoint_opts: CheckpointOpts,
    target_filename: str,
    document_index: DocumentIndex,
    payload_stream: Iterator[DocumentPayload],
    token2id: Token2Id = None,
) -> Iterable[DocumentPayload]:

    serializer: IContentSerializer = create_serializer(checkpoint_opts)

    assert_that_path_exists(path_of(target_filename))

    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:

        zf.writestr(CHECKPOINT_OPTS_FILENAME, json.dumps(asdict(checkpoint_opts)).encode('utf8'))

        for payload in payload_stream:
            data = serializer.serialize(payload.content, checkpoint_opts)
            zf.writestr(payload.filename, data=data)
            yield payload

        if document_index is not None:
            document_index_name = checkpoint_opts.document_index_name or DOCUMENT_INDEX_FILENAME
            document_index_sep = checkpoint_opts.document_index_sep or "\t"
            zf.writestr(document_index_name, data=document_index.to_csv(sep=document_index_sep, header=True))

        if token2id is not None:
            zf.writestr(DICTIONARY_FILENAME, data=json.dumps(token2id.data))


class CheckpointReader(zipfile.ZipFile):
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

        data_str = zip_utils.read_file_content(zip_or_filename=self, filename=self.document_index_name, as_binary=False)
        document_index = load_document_index(StringIO(data_str), sep=self.checkpoint_opts.document_index_sep)
        return document_index

    def _token2id(self) -> Optional[Token2Id]:
        """Returns dictionary stored in archive, or None if not found in archive"""

        if DICTIONARY_FILENAME not in self.namelist():
            return None

        _token2id_dict: dict = zip_utils.read_json(zip_or_filename=self, filename=DICTIONARY_FILENAME)
        _token2id: Token2Id = Token2Id(_token2id_dict)
        return _token2id

    @property
    def document_filenames(self) -> List[str]:
        filenames = [f for f in self.namelist() if f not in [self.document_index_name, CHECKPOINT_OPTS_FILENAME]]
        return filenames


def load_checkpoint(
    source_name: str,
    checkpoint_opts: CheckpointOpts = None,
    reader_opts: TextReaderOpts = None,
    deserialize_stream: Callable[[str, CheckpointOpts, List[str]], Iterable[DocumentPayload]] = None,
) -> CheckpointData:
    """Load a tagged frame checkpoint stored in a zipped file with CSV-filed and optionally a document index

    Currently reader_opts is only used when pandas doc index should be created (might also be used to filter files).

    Args:
        source_name (str): [description]
        options (CheckpointOpts, optional): deserialize opts. Defaults to None.
        reader_opts (TextReaderOpts, optional): Settings for creatin a document index or filtering files. Defaults to None.

    Raises:
        PipelineError: [description]

    Returns:
        CheckpointData: [description]
    """

    with CheckpointReader(source_name, mode="r", checkpoint_opts=checkpoint_opts) as zf:

        filenames: List[str] = zf.document_filenames
        document_index: DocumentIndex = zf.document_index
        checkpoint_opts: CheckpointOpts = zf.checkpoint_opts
        token2id: Token2Id = zf.token2id

        if document_index is None:
            document_index = DocumentIndexHelper.from_filenames2(filenames, reader_opts)

    if document_index is None:

        logger.warning(f"Checkpoint {source_name} has no document index (I hope you have one separately")

    elif filenames != document_index.filename.to_list():

        """ Check that filenames and document index are in sync """
        if set(filenames) != set(document_index.filename.to_list()):
            raise Exception(f"{source_name} archive filenames and document index filenames differs")

        logger.warning(f"{source_name} filename sort order mismatch (using document index sort order)")

        filenames = document_index.filename.to_list()

    if reader_opts:

        filenames = filenames_satisfied_by(
            filenames, filename_filter=reader_opts.filename_filter, filename_pattern=reader_opts.filename_pattern
        )

        if document_index is not None:
            document_index = document_index[document_index.filename.isin(filenames)]

    deserialized_stream = deserialize_stream or (
        parallel_deserialized_payload_stream if checkpoint_opts.deserialize_in_parallel else deserialized_payload_stream
    )

    create_stream = lambda: deserialized_stream(source_name, checkpoint_opts, filenames)

    data: CheckpointData = CheckpointData(
        content_type=checkpoint_opts.content_type,
        create_stream=create_stream,
        document_index=document_index,
        token2id=token2id,
        checkpoint_opts=checkpoint_opts,
        source_name=basename(source_name),
    )

    return data
