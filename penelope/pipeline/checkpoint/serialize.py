import zipfile
from io import StringIO
from multiprocessing import Pool
from typing import Any, Iterable, List, Sequence, Tuple

import pandas as pd
from penelope.utility import deprecated, getLogger

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
    logger.info("Using sequential deserialization")

    with zipfile.ZipFile(source_name, mode="r") as zf:
        for filename in filenames:
            content: str = zf.read(filename).decode(encoding='utf-8')
            tagged_frame: TaggedFrame = serializer.deserialize(content, checkpoint_opts)
            yield DocumentPayload(
                content_type=checkpoint_opts.content_type,
                content=tagged_frame,
                filename=filename,
            )


def _process_document_file(args: List[Tuple]) -> DocumentPayload:

    filename, content, serializer, checkpoint_opts = args

    return DocumentPayload(
        content_type=checkpoint_opts.content_type,
        content=serializer.deserialize(content, checkpoint_opts),
        filename=filename,
    )


def _process_document_with_read(args: Tuple) -> DocumentPayload:

    filename, source, serializer, checkpoint_opts = args

    with zipfile.ZipFile(source, mode="r") as zf:
        content = zf.read(filename).decode(encoding='utf-8')

    return DocumentPayload(
        content_type=checkpoint_opts.content_type,
        content=serializer.deserialize(content, checkpoint_opts),
        filename=filename,
    )


@deprecated
def parallel_deserialized_payload_stream_read_ahead_without_chunks(
    source_name: str, checkpoint_opts: CheckpointOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    logger.info(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
    serializer: IContentSerializer = create_serializer(checkpoint_opts)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        args: str = [
            (filename, zf.read(filename).decode(encoding='utf-8'), serializer, checkpoint_opts)
            for filename in filenames
        ]

    with Pool(processes=checkpoint_opts.deserialize_processes) as executor:
        payloads_futures: Iterable[DocumentPayload] = executor.map(_process_document_file, args)

        for payload in payloads_futures:
            yield payload


def chunker(seq: Sequence[Any], size: int) -> Sequence[Any]:
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


@deprecated
def parallel_deserialized_payload_stream_read_ahead_with_chunks(
    source_name: str,
    checkpoint_opts: CheckpointOpts,
    filenames: List[str],
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    logger.info(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
    serializer: IContentSerializer = create_serializer(checkpoint_opts)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        with Pool(processes=checkpoint_opts.deserialize_processes) as executor:
            for filenames_chunk in chunker(filenames, checkpoint_opts.deserialize_chunksize):
                args: str = [
                    (filename, zf.read(filename).decode(encoding='utf-8'), serializer, checkpoint_opts)
                    for filename in filenames_chunk
                ]
                payloads_futures: Iterable[DocumentPayload] = executor.map(_process_document_file, args)

                for payload in payloads_futures:
                    yield payload


def parallel_deserialized_payload_stream(
    source_name: str,
    checkpoint_opts: CheckpointOpts,
    filenames: List[str],
) -> Iterable[DocumentPayload]:

    logger.info(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
    serializer: IContentSerializer = create_serializer(checkpoint_opts)

    with Pool(processes=checkpoint_opts.deserialize_processes) as executor:
        args: str = [(filename, source_name, serializer, checkpoint_opts) for filename in filenames]
        payloads_futures: Iterable[DocumentPayload] = executor.imap(
            _process_document_with_read, args, chunksize=checkpoint_opts.deserialize_chunksize
        )

        for payload in payloads_futures:
            yield payload
