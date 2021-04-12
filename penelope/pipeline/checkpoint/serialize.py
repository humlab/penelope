import zipfile
from io import StringIO
from typing import Iterable, List, Sequence

import pandas as pd
from penelope.utility import getLogger

from ..interfaces import ContentType, DocumentPayload
from ..tagged_frame import TaggedFrame
from .interface import CheckpointOpts, IContentSerializer

logger = getLogger("penelope")


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


def create_serializer(options: CheckpointOpts) -> "IContentSerializer":

    if options.custom_serializer:
        return options.custom_serializer()

    if options.content_type == ContentType.TEXT:
        return TextContentSerializer()

    if options.content_type == ContentType.TOKENS:
        return TokensContentSerializer()

    if options.content_type == ContentType.TAGGED_FRAME:
        return CsvContentSerializer()

    raise ValueError(f"non-serializable content type: {options.content_type}")


def deserialized_payload_stream(
    source_name: str, checkpoint_opts: CheckpointOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    serializer: IContentSerializer = create_serializer(checkpoint_opts)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        for filename in filenames:
            content: str = zf.read(filename).decode(encoding='utf-8')
            tagged_frame: TaggedFrame = serializer.deserialize(content, checkpoint_opts)
            yield DocumentPayload(
                content_type=checkpoint_opts.content_type,
                content=tagged_frame,
                filename=filename,
            )
