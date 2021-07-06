import os
import zipfile
from collections import defaultdict
from io import StringIO
from multiprocessing import get_context
from typing import Callable, Iterable, List, Mapping, Sequence, Tuple, Union

import pandas as pd
from loguru import logger
from penelope.utility.zip_utils import zipfile_or_filename
from tqdm import tqdm

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


@zipfile_or_filename(mode='r')
def read_and_deserialize_tagged_frame(
    *,
    zip_or_filename: Union[str, zipfile.ZipFile],
    filename: str,
    checkpoint_opts: CheckpointOpts,
    serializer: Serializer,
) -> TaggedFrame:
    content: str = zip_or_filename.read(filename).decode(encoding='utf-8')
    tagged_frame: TaggedFrame = serializer.deserialize(content, checkpoint_opts)
    return tagged_frame


def read_and_deserialize_tagged_frame_with_feather_cache(
    *,
    zip_or_filename: Union[str, zipfile.ZipFile],
    filename: str,
    checkpoint_opts: CheckpointOpts,
    serializer: Serializer,
) -> TaggedFrame:
    feather_filename: str = checkpoint_opts.feather_filename(filename)
    if os.path.isfile(feather_filename):
        tagged_frame: pd.DataFrame = pd.read_feather(feather_filename)
    else:
        tagged_frame = read_and_deserialize_tagged_frame(
            zip_or_filename=zip_or_filename,
            filename=filename,
            checkpoint_opts=checkpoint_opts,
            serializer=serializer,
        )
        tagged_frame.to_feather(feather_filename, compression="lz4")
    return tagged_frame


# FIXME: This only works if input is tagged frame (not tokens, text)


def sequential_deserialized_payload_stream(
    *,
    zip_or_filename: Union[str, zipfile.ZipFile],
    checkpoint_opts: CheckpointOpts,
    filenames: List[str],
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    serializer: IContentSerializer = create_serializer(checkpoint_opts)
    logger.debug("Using sequential deserialization")

    if checkpoint_opts.feather_folder:
        logger.debug(f"Using feather checkpoint folder {checkpoint_opts.feather_folder}.")
        reader = read_and_deserialize_tagged_frame_with_feather_cache
    else:
        reader = read_and_deserialize_tagged_frame

    for filename in filenames:
        tagged_frame = reader(
            zip_or_filename=zip_or_filename,
            filename=filename,
            checkpoint_opts=checkpoint_opts,
            serializer=serializer,
        )
        yield DocumentPayload(
            content_type=checkpoint_opts.content_type,
            content=tagged_frame,
            filename=filename,
        )


def _count_column_frequencies(series: pd.DataFrame) -> Mapping[str, int]:
    counts = defaultdict(int)
    for v in series:
        counts[v] += 1
    return counts


def open_read_and_deserialize_tagged_frame(args: Tuple) -> DocumentPayload:
    try:
        filename, source, serializer, checkpoint_opts = args
        tagged_frame = read_and_deserialize_tagged_frame(
            zip_or_filename=source,
            filename=filename,
            checkpoint_opts=checkpoint_opts,
            serializer=serializer,
        )
        payload = DocumentPayload(content_type=checkpoint_opts.content_type, content=tagged_frame, filename=filename)

        if checkpoint_opts.frequency_column:
            payload.remember(term_frequency=_count_column_frequencies(tagged_frame[checkpoint_opts.frequency_column]))
            payload.remember(pos_frequency=_count_column_frequencies(tagged_frame[checkpoint_opts.pos_column]))

        return payload
    except Exception as ex:
        logger.exception(ex)
        raise ex
        # return ex


def open_read_and_deserialize_tagged_frame_with_feather_cache(args: Tuple) -> DocumentPayload:
    filename, source, serializer, checkpoint_opts = args
    try:
        tagged_frame: pd.DataFrame = read_and_deserialize_tagged_frame_with_feather_cache(
            zip_or_filename=source,
            filename=filename,
            checkpoint_opts=checkpoint_opts,
            serializer=serializer,
        )

        payload = DocumentPayload(content_type=checkpoint_opts.content_type, content=tagged_frame, filename=filename)

        if checkpoint_opts.frequency_column:
            payload.remember(term_frequency=_count_column_frequencies(tagged_frame[checkpoint_opts.frequency_column]))
            payload.remember(pos_frequency=_count_column_frequencies(tagged_frame[checkpoint_opts.pos_column]))

        return payload
    except Exception as ex:
        logger.error(f"Filename: {filename}")
        logger.exception(ex)
        raise ex
        # return ex


def parallel_deserialized_payload_stream(
    *,
    zip_or_filename: str,
    checkpoint_opts: CheckpointOpts,
    filenames: List[str],
) -> Iterable[DocumentPayload]:

    try:
        logger.trace(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
        _process_document = open_read_and_deserialize_tagged_frame
        if checkpoint_opts.feather_folder:
            logger.trace(f"Using feather checkpoint folder {checkpoint_opts.feather_folder}.")
            _process_document = open_read_and_deserialize_tagged_frame_with_feather_cache

        serializer: IContentSerializer = create_serializer(checkpoint_opts)

        with get_context("spawn").Pool(processes=checkpoint_opts.deserialize_processes) as executor:
            args: Iterable[Tuple] = tqdm(
                [(filename, zip_or_filename, serializer, checkpoint_opts) for filename in filenames]
            )
            payloads_futures: Iterable[DocumentPayload] = executor.imap_unordered(
                _process_document, args, chunksize=checkpoint_opts.deserialize_chunksize
            )

            for payload in payloads_futures:
                yield payload

    except Exception as ex:
        logger.exception(ex)
        raise ex
        # return ex
