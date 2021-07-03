import os
import zipfile
from io import StringIO
from multiprocessing import get_context
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import pandas as pd
from loguru import logger

from ..interfaces import ContentType, DocumentPayload
from ..tagged_frame import TaggedFrame
from .interface import CheckpointOpts, IContentSerializer


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


Serializer = Callable[[str, CheckpointOpts], pd.DataFrame]


def read_and_deserialize_tagged_frame(
    source: Union[str, zipfile.ZipFile], filename: str, checkpoint_opts: CheckpointOpts, serializer: Serializer
) -> TaggedFrame:
    content: str = source.read(filename).decode(encoding='utf-8') if isinstance(source, zipfile.ZipFile) else source
    tagged_frame: TaggedFrame = serializer.deserialize(content, checkpoint_opts)
    return tagged_frame


def read_and_deserialize_tagged_frame_with_feather_cache(
    zf: zipfile.ZipFile, filename: str, checkpoint_opts: CheckpointOpts, serializer: Serializer
) -> TaggedFrame:
    feather_filename: str = checkpoint_opts.feather_filename(filename)
    if os.path.isfile(feather_filename):
        tagged_frame: pd.DataFrame = pd.read_feather(feather_filename)
    else:
        tagged_frame = read_and_deserialize_tagged_frame(zf, filename, checkpoint_opts, serializer)
        tagged_frame.to_feather(feather_filename, compression="lz4")
    return tagged_frame


# FIXME: This only works if input is tagged frame (not tokens, text)


def sequential_deserialized_payload_stream(
    source_name: str, checkpoint_opts: CheckpointOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    serializer: IContentSerializer = create_serializer(checkpoint_opts)
    logger.debug("Using sequential deserialization")

    if checkpoint_opts.feather_folder:
        logger.debug(f"Using feather checkpoint folder {checkpoint_opts.feather_folder}.")
        reader = read_and_deserialize_tagged_frame_with_feather_cache
    else:
        reader = read_and_deserialize_tagged_frame

    with zipfile.ZipFile(source_name, mode="r") as zf:
        for filename in filenames:
            tagged_frame = reader(zf, filename, checkpoint_opts, serializer)
            yield DocumentPayload(
                content_type=checkpoint_opts.content_type,
                content=tagged_frame,
                filename=filename,
            )


def open_read_and_deserialize_tagged_frame(args: Tuple) -> DocumentPayload:
    filename, source, serializer, checkpoint_opts = args
    with zipfile.ZipFile(source, mode="r") as zf:
        tagged_frame = read_and_deserialize_tagged_frame(zf, filename, checkpoint_opts, serializer)
    return DocumentPayload(content_type=checkpoint_opts.content_type, content=tagged_frame, filename=filename)


def open_read_and_deserialize_tagged_frame_with_feather_cache(args: Tuple) -> DocumentPayload:
    filename, source, serializer, checkpoint_opts = args
    with zipfile.ZipFile(source, mode="r") as zf:
        tagged_frame: pd.DataFrame = read_and_deserialize_tagged_frame_with_feather_cache(
            zf, filename, checkpoint_opts, serializer
        )
    return DocumentPayload(content_type=checkpoint_opts.content_type, content=tagged_frame, filename=filename)


def parallel_deserialized_payload_stream(
    source_name: str,
    checkpoint_opts: CheckpointOpts,
    filenames: List[str],
) -> Iterable[DocumentPayload]:

    logger.trace(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
    _process_document = open_read_and_deserialize_tagged_frame
    if checkpoint_opts.feather_folder:
        logger.trace(f"Using feather checkpoint folder {checkpoint_opts.feather_folder}.")
        _process_document = open_read_and_deserialize_tagged_frame_with_feather_cache

    serializer: IContentSerializer = create_serializer(checkpoint_opts)

    with get_context("spawn").Pool(processes=checkpoint_opts.deserialize_processes) as executor:
        args: str = [(filename, source_name, serializer, checkpoint_opts) for filename in filenames]
        payloads_futures: Iterable[DocumentPayload] = executor.imap(
            _process_document, args, chunksize=checkpoint_opts.deserialize_chunksize
        )

        for payload in payloads_futures:
            yield payload
