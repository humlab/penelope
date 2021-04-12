import abc
import copy
import csv
import json
import zipfile
from dataclasses import asdict, dataclass, field
from io import StringIO
from os.path import basename
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Union

import pandas as pd
from penelope.corpus import DocumentIndex, DocumentIndexHelper, TextReaderOpts, load_document_index
from penelope.utility import (
    assert_that_path_exists,
    create_instance,
    filenames_satisfied_by,
    getLogger,
    path_of,
    zip_utils,
)

from .interfaces import ContentType, DocumentPayload, PipelineError
from .tagged_frame import TaggedFrame

SerializableContent = Union[str, Iterable[str], TaggedFrame]

CHECKPOINT_OPTS_FILENAME = "options.json"
DOCUMENT_INDEX_FILENAME = "document_index.csv"

logger = getLogger("penelope")


@dataclass
class CheckpointOpts:

    content_type_code: int = 0

    document_index_name: str = field(default="document_index.csv")
    document_index_sep: str = field(default='\t')

    sep: str = '\t'
    quoting: int = csv.QUOTE_NONE
    custom_serializer_classname: str = None

    text_column: str = field(default="text")
    lemma_column: str = field(default="lemma")
    pos_column: str = field(default="pos")
    extra_columns: List[str] = field(default_factory=list)
    index_column: Union[int, None] = 0

    @property
    def content_type(self) -> ContentType:
        return ContentType(self.content_type_code)

    @content_type.setter
    def content_type(self, value: ContentType):
        self.content_type_code = int(value)

    def as_type(self, value: ContentType) -> "CheckpointOpts":
        # FIXME #45 Not all member properties are copies in type cast
        opts = copy.copy(self)
        opts.content_type_code = int(value)
        return opts

    @staticmethod
    def load(data: dict) -> "CheckpointOpts":
        opts = CheckpointOpts()
        for key in data.keys():
            if hasattr(opts, key):
                setattr(opts, key, data[key])
        return opts

    @property
    def custom_serializer(self) -> type:
        if not self.custom_serializer_classname:
            return None
        return create_instance(self.custom_serializer_classname)

    @property
    def columns(self) -> List[str]:
        return [self.text_column, self.lemma_column, self.pos_column] + (self.extra_columns or [])

    def text_column_name(self, lemmatized: bool = False):
        return self.lemma_column if lemmatized else self.text_column


@dataclass
class CheckpointData:
    source_name: Any = None
    content_type: ContentType = ContentType.NONE
    document_index: DocumentIndex = None
    payload_stream: Iterable[DocumentPayload] = None
    checkpoint_opts: CheckpointOpts = None


class IContentSerializer(abc.ABC):
    @abc.abstractmethod
    def serialize(self, content: SerializableContent, options: CheckpointOpts) -> str:
        ...

    @abc.abstractmethod
    def deserialize(self, content: str, options: CheckpointOpts) -> SerializableContent:
        ...

    @staticmethod
    def create(options: CheckpointOpts) -> "IContentSerializer":

        if options.custom_serializer:
            return options.custom_serializer()

        if options.content_type == ContentType.TEXT:
            return TextContentSerializer()

        if options.content_type == ContentType.TOKENS:
            return TokensContentSerializer()

        if options.content_type == ContentType.TAGGED_FRAME:
            return CsvContentSerializer()

        raise ValueError(f"non-serializable content type: {options.content_type}")


class TextContentSerializer(IContentSerializer):
    def serialize(self, content: str, options: CheckpointOpts) -> str:
        return content

    def deserialize(self, content: str, options: CheckpointOpts) -> str:
        return content


class TokensContentSerializer(IContentSerializer):
    def serialize(self, content: Sequence[str], options: CheckpointOpts) -> str:
        return ' '.join(content)

    def deserialize(self, content: str, options: CheckpointOpts) -> Sequence[str]:
        return content.split(' ')


class CsvContentSerializer(IContentSerializer):
    def serialize(self, content: pd.DataFrame, options: CheckpointOpts) -> str:
        return content.to_csv(sep=options.sep, header=True)

    def deserialize(self, content: str, options: CheckpointOpts) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_csv(
            StringIO(content), sep=options.sep, quoting=options.quoting, index_col=options.index_column
        )
        data.fillna("", inplace=True)
        if any(x not in data.columns for x in options.columns):
            raise ValueError(f"missing columns: {', '.join([x for x in options.columns if x not in data.columns])}")
        return data[options.columns]


def store_checkpoint(
    *,
    checkpoint_opts: CheckpointOpts,
    target_filename: str,
    document_index: DocumentIndex,
    payload_stream: Iterator[DocumentPayload],
) -> Iterable[DocumentPayload]:

    serializer: IContentSerializer = IContentSerializer.create(checkpoint_opts)

    assert_that_path_exists(path_of(target_filename))

    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:

        zf.writestr(CHECKPOINT_OPTS_FILENAME, json.dumps(asdict(checkpoint_opts)).encode('utf8'))

        for payload in payload_stream:
            data = serializer.serialize(payload.content, checkpoint_opts)
            zf.writestr(payload.filename, data=data)
            yield payload

        if document_index is not None:
            zf.writestr(
                checkpoint_opts.document_index_name,
                data=document_index.to_csv(sep=checkpoint_opts.document_index_sep, header=True),
            )


class CheckpointZipFile(zipfile.ZipFile):
    def __init__(
        self,
        file: Any,
        mode: str,
        checkpoint_opts: CheckpointOpts = None,
        document_index_name=DOCUMENT_INDEX_FILENAME,
    ) -> None:

        super().__init__(file, mode=mode, compresslevel=zipfile.ZIP_DEFLATED)

        self.checkpoint_opts = checkpoint_opts or self._checkpoint_opts()

        if self.checkpoint_opts is None:
            raise PipelineError(
                f"Checkpoint options not supplied and file {CHECKPOINT_OPTS_FILENAME} not found in archive."
            )

        self.document_index_name = document_index_name or self.checkpoint_opts.document_index_name
        self.document_index = self._document_index()

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

        data_str = zip_utils.read(zip_or_filename=self, filename=self.document_index_name, as_binary=False)
        document_index = load_document_index(StringIO(data_str), sep=self.checkpoint_opts.document_index_sep)
        return document_index

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
    deserialized_stream = deserialize_stream or deserialized_payload_stream
    with CheckpointZipFile(source_name, mode="r", checkpoint_opts=checkpoint_opts) as zf:

        filenames: List[str] = zf.document_filenames
        document_index: DocumentIndex = zf.document_index
        checkpoint_opts: CheckpointOpts = zf.checkpoint_opts

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

    payload_stream = deserialized_stream(source_name, checkpoint_opts, filenames)

    data: CheckpointData = CheckpointData(
        content_type=checkpoint_opts.content_type,
        payload_stream=payload_stream,
        document_index=document_index,
        checkpoint_opts=checkpoint_opts,
        source_name=basename(source_name),
    )

    return data


def deserialized_payload_stream(
    source_name: str, checkpoint_opts: CheckpointOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    serializer: IContentSerializer = IContentSerializer.create(checkpoint_opts)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        for filename in filenames:
            content: str = zf.read(filename).decode(encoding='utf-8')
            tagged_frame: TaggedFrame = serializer.deserialize(content, checkpoint_opts)
            yield DocumentPayload(
                content_type=checkpoint_opts.content_type,
                content=tagged_frame,  # serializer.deserialize(content, checkpoint_opts),
                filename=filename,
            )
