import json
import os
import zipfile
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Any, Callable, Dict, Iterable, Iterator, Union

import pandas as pd
from penelope.pipeline.utils import load_document_index

from . import interfaces

SerializableContent = Union[str, Iterable[str], pd.core.api.DataFrame]


@dataclass
class ContentSerializer:

    serialize: Callable[[SerializableContent], str] = None
    deserialize: Callable[[str], SerializableContent] = None

    @staticmethod
    def identity(content: Any):
        return content

    @staticmethod
    def token_to_text(content: Iterable[str]) -> str:
        return ' '.join(content)

    @staticmethod
    def text_to_token(content: str) -> Iterable[str]:
        return content.split(' ')

    @staticmethod
    def df_to_text(content: pd.DataFrame) -> str:
        return content.to_csv(sep='\t', header=True)

    @staticmethod
    def text_to_df(content: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(content), sep='\t', index_col=0)

    @staticmethod
    def read_text(zf: zipfile.ZipFile, filename: str) -> str:
        return zf.read(filename).decode('utf-8')

    @staticmethod
    def read_binary(zf: zipfile.ZipFile, filename: str) -> bytes:
        return zf.read(filename)

    @staticmethod
    def read_json(zf: zipfile.ZipFile, filename: str) -> Dict:
        return json.loads(ContentSerializer.read_text(zf, filename))

    @staticmethod
    def read_dataframe(zf: zipfile.ZipFile, filename: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(ContentSerializer.read_text(zf, filename)), sep='\t', index_col=0)


CHECKPOINT_SERIALIZERS = {
    interfaces.ContentType.TEXT: ContentSerializer(
        serialize=ContentSerializer.identity, deserialize=ContentSerializer.identity
    ),
    interfaces.ContentType.TOKENS: ContentSerializer(
        serialize=ContentSerializer.token_to_text, deserialize=ContentSerializer.text_to_token
    ),
    interfaces.ContentType.TAGGEDFRAME: ContentSerializer(
        serialize=ContentSerializer.df_to_text, deserialize=ContentSerializer.text_to_df
    ),
    # FIXME: ADD SPARV XML with as_binary
}


@dataclass
class ContentSerializeOpts:
    content_type_code: int = 0
    document_index_name: str = field(default="document_index.csv")
    as_binary: bool = False

    @property
    def content_type(self) -> interfaces.ContentType:
        return interfaces.ContentType(self.content_type_code)

    @content_type.setter
    def content_type(self, value: interfaces.ContentType):
        self.content_type_code = int(value)


@dataclass
class CheckpointData:
    content_type: interfaces.ContentType = interfaces.ContentType.NONE
    document_index: pd.DataFrame = None
    payload_stream: Iterable[interfaces.DocumentPayload] = None
    serialize_opts: ContentSerializeOpts = None


def store_checkpoint(
    *,
    options: ContentSerializeOpts,
    target_filename: str,
    document_index: pd.DataFrame,
    payload_stream: Iterator[interfaces.DocumentPayload],
) -> Iterable[interfaces.DocumentPayload]:

    serializer = CHECKPOINT_SERIALIZERS[options.content_type]

    store_path = os.path.split(target_filename)[0]
    if not os.path.isdir(store_path):
        raise FileNotFoundError(f"target folder {store_path} does not exist")

    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:

        zf.writestr("options.json", json.dumps(asdict(options)).encode('utf8'))

        if document_index is not None:
            zf.writestr(options.document_index_name, data=document_index.to_csv(sep='\t', header=True))

        for payload in payload_stream:
            zf.writestr(payload.filename, data=serializer.serialize(payload.content))
            yield payload


def load_checkpoint(source_filename: str, document_index_key_column: str) -> CheckpointData:

    with zipfile.ZipFile(source_filename, mode="r") as zf:

        filenames = zf.namelist()

        if "options.json" not in filenames:
            raise interfaces.PipelineError("Checkpoint file is not valid (has no options.json")

        serialize_opts = ContentSerializeOpts(**ContentSerializer.read_json(zf, "options.json"))

        document_index = None
        if serialize_opts.document_index_name in filenames:
            data_str = ContentSerializer.read_text(zf, serialize_opts.document_index_name)
            document_index = load_document_index(StringIO(data_str), key_column=document_index_key_column, sep='\t')
            filenames.remove(serialize_opts.document_index_name)

        filenames.remove("options.json")

    content_reader = ContentSerializer.read_binary if serialize_opts.as_binary else ContentSerializer.read_text
    serializer = CHECKPOINT_SERIALIZERS[serialize_opts.content_type]

    def payload_stream():
        with zipfile.ZipFile(source_filename, mode="r") as zf:
            for filename in filenames:
                yield interfaces.DocumentPayload(
                    content_type=serialize_opts.content_type,
                    content=serializer.deserialize(content_reader(zf, filename)),
                    filename=filename,
                )

    data: CheckpointData = CheckpointData(
        content_type=serialize_opts.content_type,
        payload_stream=payload_stream(),
        document_index=document_index,
        serialize_opts=serialize_opts,
    )

    return data
