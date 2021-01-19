import abc
import json
import zipfile
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Iterable, Iterator, List, Sequence, Union

import pandas as pd
import penelope.utility.zip_utils as zip_utils
from penelope.corpus import DocumentIndex, load_document_index
from penelope.corpus.readers.interfaces import TextReaderOpts
from penelope.utility import assert_that_path_exists, getLogger, path_of

from .config import CorpusSerializeOpts
from .interfaces import ContentType, DocumentPayload, PipelineError

SerializableContent = Union[str, Iterable[str], pd.core.api.DataFrame]

logger = getLogger("penelope")


@dataclass
class CheckpointData:
    content_type: ContentType = ContentType.NONE
    document_index: pd.DataFrame = None
    payload_stream: Iterable[DocumentPayload] = None
    serialize_opts: CorpusSerializeOpts = None


class IContentSerializer(abc.ABC):
    @abc.abstractmethod
    def serialize(self, content: SerializableContent, options: CorpusSerializeOpts) -> str:
        ...

    @abc.abstractmethod
    def deserialize(self, content: str, options: CorpusSerializeOpts) -> SerializableContent:
        ...

    @staticmethod
    def create(content_type: ContentType) -> "IContentSerializer":

        if content_type == ContentType.TEXT:
            return TextContentSerializer()

        if content_type == ContentType.TOKENS:
            return TokensContentSerializer()

        if content_type == ContentType.TAGGEDFRAME:
            return TaggedFrameContentSerializer()

        raise ValueError(f"non-serializable content type: {content_type}")


class TextContentSerializer(IContentSerializer):
    def serialize(self, content: str, options: CorpusSerializeOpts) -> str:
        return content

    def deserialize(self, content: str, options: CorpusSerializeOpts) -> str:
        return content


class TokensContentSerializer(IContentSerializer):
    def serialize(self, content: Sequence[str], options: CorpusSerializeOpts) -> str:
        return ' '.join(content)

    def deserialize(self, content: str, options: CorpusSerializeOpts) -> Sequence[str]:
        return content.split(' ')


class TaggedFrameContentSerializer(IContentSerializer):
    def serialize(self, content: pd.DataFrame, options: CorpusSerializeOpts) -> str:
        return content.to_csv(sep=options.sep, header=True)

    def deserialize(self, content: str, options: CorpusSerializeOpts) -> pd.DataFrame:
        return pd.read_csv(StringIO(content), sep=options.sep, quoting=options.quoting, index_col=0)


SERIALIZE_OPT_FILENAME = "options.json"


def store_checkpoint(
    *,
    options: CorpusSerializeOpts,
    target_filename: str,
    document_index: pd.DataFrame,
    payload_stream: Iterator[DocumentPayload],
) -> Iterable[DocumentPayload]:

    serializer: IContentSerializer = IContentSerializer.create(options.content_type)

    assert_that_path_exists(path_of(target_filename))

    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:

        zf.writestr(SERIALIZE_OPT_FILENAME, json.dumps(asdict(options)).encode('utf8'))

        for payload in payload_stream:
            data = serializer.serialize(payload.content, options)
            zf.writestr(payload.filename, data=data)
            yield payload

        if document_index is not None:
            zf.writestr(
                options.document_index_name,
                data=document_index.to_csv(sep=options.document_index_sep, header=True),
            )


def load_checkpoint(
    source_name: str, options: CorpusSerializeOpts = None, reader_opts: TextReaderOpts = None
) -> CheckpointData:

    # FIXME: Currently reader_opts is only used when pandas doc index should be created.  Might also be used to filter files.
    with zipfile.ZipFile(source_name, mode="r") as zf:

        filenames = zf.namelist()

        if options is None:

            if SERIALIZE_OPT_FILENAME not in filenames:
                raise PipelineError("options not supplied and not found in archive (missing options.json)")

            stored_opts = zip_utils.read_json(zip_or_filename=zf, filename=SERIALIZE_OPT_FILENAME)
            options = CorpusSerializeOpts.load(stored_opts)

            filenames.remove(SERIALIZE_OPT_FILENAME)

        document_index = None

        if options.document_index_name and options.document_index_name in filenames:

            data_str = zip_utils.read(zip_or_filename=zf, filename=options.document_index_name, as_binary=False)
            document_index = load_document_index(StringIO(data_str), sep=options.document_index_sep)

            filenames.remove(options.document_index_name)

        elif reader_opts and reader_opts.filename_fields is not None:
            document_index = DocumentIndex.from_filenames(
                filenames=filenames,
                filename_fields=reader_opts.filename_fields,
            ).document_index

    data: CheckpointData = CheckpointData(
        content_type=options.content_type,
        payload_stream=deserialized_payload_stream(source_name, options, filenames),
        document_index=document_index,
        serialize_opts=options,
    )

    return data


def deserialized_payload_stream(
    source_name: str, options: CorpusSerializeOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    serializer: IContentSerializer = IContentSerializer.create(options.content_type)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        for filename in filenames:
            content = zip_utils.read(zip_or_filename=zf, filename=filename, as_binary=False)
            yield DocumentPayload(
                content_type=options.content_type,
                content=serializer.deserialize(content, options),
                filename=filename,
            )
