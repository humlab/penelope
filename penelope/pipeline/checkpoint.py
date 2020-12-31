import json
import zipfile
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Any, Callable, Dict, Iterable, Iterator, List, Union

import pandas as pd
from penelope.corpus import DocumentIndex, load_document_index
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


@dataclass
class ContentSerializer:

    serialize: Callable[[SerializableContent, CorpusSerializeOpts], str] = None
    deserialize: Callable[[str, CorpusSerializeOpts], SerializableContent] = None

    @staticmethod
    def identity(content: Any, _: CorpusSerializeOpts):
        return content

    @staticmethod
    def token_to_text(content: Iterable[str], _: CorpusSerializeOpts) -> str:
        return ' '.join(content)

    @staticmethod
    def text_to_token(content: str, _: CorpusSerializeOpts) -> Iterable[str]:
        return content.split(' ')

    @staticmethod
    def df_to_text(content: pd.DataFrame, opts: CorpusSerializeOpts) -> str:
        return content.to_csv(sep=opts.sep, header=True)

    @staticmethod
    def text_to_df(content: str, opts: CorpusSerializeOpts) -> pd.DataFrame:
        return pd.read_csv(StringIO(content), sep=opts.sep, quoting=opts.quoting, index_col=0)

    @staticmethod
    def read_text(zf: zipfile.ZipFile, filename: str, _: CorpusSerializeOpts) -> str:
        return zf.read(filename).decode('utf-8')

    @staticmethod
    def read_binary(zf: zipfile.ZipFile, filename: str, _: CorpusSerializeOpts) -> bytes:
        return zf.read(filename)

    @staticmethod
    def read_json(zf: zipfile.ZipFile, filename: str, opts: CorpusSerializeOpts) -> Dict:
        return json.loads(ContentSerializer.read_text(zf, filename, opts))

    @staticmethod
    def read_dataframe(zf: zipfile.ZipFile, filename: str, opts: CorpusSerializeOpts) -> pd.DataFrame:
        data_str = ContentSerializer.read_text(zf, filename, opts)
        df = pd.read_csv(StringIO(data_str), sep=opts.sep, quoting=opts.quoting, index_col=0)
        return df

    @staticmethod
    def create(content_type: ContentType) -> "ContentSerializer":

        if content_type == ContentType.TEXT:
            return ContentSerializer(
                serialize=ContentSerializer.identity,
                deserialize=ContentSerializer.identity,
            )

        if content_type == ContentType.TOKENS:
            return ContentSerializer(
                serialize=ContentSerializer.token_to_text,
                deserialize=ContentSerializer.text_to_token,
            )

        if content_type == ContentType.TAGGEDFRAME:
            return ContentSerializer(
                serialize=ContentSerializer.df_to_text,
                deserialize=ContentSerializer.text_to_df,
            )

        raise ValueError(f"non-serializable content type: {content_type}")


SERIALIZE_OPT_FILENAME = "options.json"


def store_checkpoint(
    *,
    options: CorpusSerializeOpts,
    target_filename: str,
    document_index: pd.DataFrame,
    payload_stream: Iterator[DocumentPayload],
) -> Iterable[DocumentPayload]:

    serializer: ContentSerializer = ContentSerializer.create(options.content_type)

    assert_that_path_exists(path_of(target_filename))

    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:

        zf.writestr(SERIALIZE_OPT_FILENAME, json.dumps(asdict(options)).encode('utf8'))

        for payload in payload_stream:
            zf.writestr(payload.filename, data=serializer.serialize(payload.content, options))
            yield payload

        if document_index is not None:
            zf.writestr(
                options.document_index_name,
                data=document_index.to_csv(sep=options.document_index_sep, header=True),
            )


def load_checkpoint(source_name: str, options: CorpusSerializeOpts = None) -> CheckpointData:

    with zipfile.ZipFile(source_name, mode="r") as zf:

        filenames = zf.namelist()

        if options is None:

            if SERIALIZE_OPT_FILENAME not in filenames:
                raise PipelineError("options not supplied and not found in archive (missing options.json)")

            options = CorpusSerializeOpts(**ContentSerializer.read_json(zf, SERIALIZE_OPT_FILENAME, options))

            filenames.remove(SERIALIZE_OPT_FILENAME)

        document_index = None

        if options.document_index_name and options.document_index_name in filenames:
            data_str = ContentSerializer.read_text(zf, options.document_index_name, options)
            document_index = load_document_index(
                StringIO(data_str),
                sep=options.sep,
            )

            filenames.remove(options.document_index_name)

        elif options.reader_opts and options.reader_opts.filename_fields is not None:
            document_index = DocumentIndex.from_filenames(
                filenames=filenames,
                filename_fields=options.reader_opts.filename_fields,
            )

    data: CheckpointData = CheckpointData(
        content_type=options.content_type,
        payload_stream=_deserialized_payload_stream(source_name, options, filenames),
        document_index=document_index,
        serialize_opts=options,
    )

    return data


def _deserialized_payload_stream(
    source_name: str, options: CorpusSerializeOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    reader = ContentSerializer.read_binary if options.as_binary else ContentSerializer.read_text
    serializer: ContentSerializer = ContentSerializer.create(options.content_type)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        for filename in filenames:
            yield DocumentPayload(
                content_type=options.content_type,
                content=serializer.deserialize(reader(zf, filename, options), options),
                filename=filename,
            )
