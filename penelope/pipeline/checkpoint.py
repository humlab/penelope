import abc
import csv
import json
import zipfile
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Iterable, Iterator, List, Sequence, Union

import pandas as pd
from penelope.corpus import DocumentIndex, DocumentIndexHelper, load_document_index
from penelope.corpus.readers import TextReaderOpts
from penelope.utility import assert_that_path_exists, create_instance, getLogger, path_of, zip_utils

from . import ContentType, DocumentPayload, PipelineError
from .tagged_frame import TaggedFrame

SerializableContent = Union[str, Iterable[str], TaggedFrame]
SERIALIZE_OPT_FILENAME = "options.json"

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

    @property
    def content_type(self) -> ContentType:
        return ContentType(self.content_type_code)

    @content_type.setter
    def content_type(self, value: ContentType):
        self.content_type_code = int(value)

    def as_type(self, value: ContentType) -> "CheckpointOpts":
        opts = CheckpointOpts(
            content_type_code=int(value),
            document_index_name=self.document_index_name,
            document_index_sep=self.document_index_sep,
            sep=self.sep,
            quoting=self.quoting,
        )
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

@dataclass
class CheckpointData:
    content_type: ContentType = ContentType.NONE
    document_index: DocumentIndex = None
    payload_stream: Iterable[DocumentPayload] = None
    serialize_opts: CheckpointOpts = None


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
            return TaggedFrameContentSerializer()

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


class TaggedFrameContentSerializer(IContentSerializer):
    def serialize(self, content: pd.DataFrame, options: CheckpointOpts) -> str:
        return content.to_csv(sep=options.sep, header=True)

    def deserialize(self, content: str, options: CheckpointOpts) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_csv(StringIO(content), sep=options.sep, quoting=options.quoting, index_col=0)
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

        zf.writestr(SERIALIZE_OPT_FILENAME, json.dumps(asdict(checkpoint_opts)).encode('utf8'))

        for payload in payload_stream:
            data = serializer.serialize(payload.content, checkpoint_opts)
            zf.writestr(payload.filename, data=data)
            yield payload

        if document_index is not None:
            zf.writestr(
                checkpoint_opts.document_index_name,
                data=document_index.to_csv(sep=checkpoint_opts.document_index_sep, header=True),
            )


def load_checkpoint(
    source_name: str, checkpoint_opts: CheckpointOpts = None, reader_opts: TextReaderOpts = None
) -> CheckpointData:
    """Load a tagged frame checkpoint stored in a zipped file with CSV-filed and optionally a document index

    Currently reader_opts is only used when pandas doc index should be created (might also be used to filter files).

    Args:
        source_name (str): [description]
        options (CheckpointOpts, optional): Deserialize oprs. Defaults to None.
        reader_opts (TextReaderOpts, optional): Create document index options (if specified). Defaults to None.

    Raises:
        PipelineError: [description]

    Returns:
        CheckpointData: [description]
    """
    with zipfile.ZipFile(source_name, mode="r") as zf:

        filenames = zf.namelist()

        if checkpoint_opts is None:

            if SERIALIZE_OPT_FILENAME not in filenames:
                raise PipelineError("options not supplied and not found in archive (missing options.json)")

            stored_opts = zip_utils.read_json(zip_or_filename=zf, filename=SERIALIZE_OPT_FILENAME)
            checkpoint_opts = CheckpointOpts.load(stored_opts)

            filenames.remove(SERIALIZE_OPT_FILENAME)

        document_index = None

        if checkpoint_opts.document_index_name and checkpoint_opts.document_index_name in filenames:

            data_str = zip_utils.read(zip_or_filename=zf, filename=checkpoint_opts.document_index_name, as_binary=False)
            document_index = load_document_index(StringIO(data_str), sep=checkpoint_opts.document_index_sep)

            filenames.remove(checkpoint_opts.document_index_name)

        elif reader_opts and reader_opts.filename_fields is not None:
            document_index = DocumentIndexHelper.from_filenames(
                filenames=filenames,
                filename_fields=reader_opts.filename_fields,
            ).document_index

    data: CheckpointData = CheckpointData(
        content_type=checkpoint_opts.content_type,
        payload_stream=deserialized_payload_stream(source_name, checkpoint_opts, filenames),
        document_index=document_index,
        serialize_opts=checkpoint_opts,
    )

    return data


def deserialized_payload_stream(
    source_name: str, checkpoint_opts: CheckpointOpts, filenames: List[str],
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    serializer: IContentSerializer = IContentSerializer.create(checkpoint_opts)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        for filename in filenames:
            content: str = zip_utils.read(zip_or_filename=zf, filename=filename, as_binary=False)
            yield DocumentPayload(
                content_type=checkpoint_opts.content_type,
                content=serializer.deserialize(content, checkpoint_opts),
                filename=filename,
            )
