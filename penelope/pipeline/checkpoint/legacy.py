# import zipfile
# from multiprocessing import Pool
# from typing import Any, Iterable, List, Sequence, Tuple

# from loguru import logger
# from penelope.utility import deprecated

# from ..interfaces import DocumentPayload
# from .interface import CheckpointOpts, IContentSerializer
# from .serialize import create_serializer


# def chunker(seq: Sequence[Any], size: int) -> Sequence[Any]:
#     return (seq[pos : pos + size] for pos in range(0, len(seq), size))


# @deprecated
# def _process_document_content(args: List[Tuple]) -> DocumentPayload:

#     filename, content, serializer, checkpoint_opts = args

#     return DocumentPayload(
#         content_type=checkpoint_opts.content_type,
#         content=serializer.deserialize(content, checkpoint_opts),
#         filename=filename,
#     )


# @deprecated
# def parallel_deserialized_payload_stream_read_ahead_without_chunks(
#     source_name: str, checkpoint_opts: CheckpointOpts, filenames: List[str]
# ) -> Iterable[DocumentPayload]:
#     """Yields a deserialized payload stream read from given source"""

#     logger.info(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
#     serializer: IContentSerializer = create_serializer(checkpoint_opts)

#     with zipfile.ZipFile(source_name, mode="r") as zf:
#         args: str = [
#             (filename, zf.read(filename).decode(encoding='utf-8'), serializer, checkpoint_opts)
#             for filename in filenames
#         ]

#     with Pool(processes=checkpoint_opts.deserialize_processes) as executor:
#         payloads_futures: Iterable[DocumentPayload] = executor.map(_process_document_content, args)

#         for payload in payloads_futures:
#             yield payload


# @deprecated
# def parallel_deserialized_payload_stream_read_ahead_with_chunks(
#     source_name: str,
#     checkpoint_opts: CheckpointOpts,
#     filenames: List[str],
# ) -> Iterable[DocumentPayload]:
#     """Yields a deserialized payload stream read from given source"""

#     logger.info(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
#     serializer: IContentSerializer = create_serializer(checkpoint_opts)

#     with zipfile.ZipFile(source_name, mode="r") as zf:
#         with Pool(processes=checkpoint_opts.deserialize_processes) as executor:
#             for filenames_chunk in chunker(filenames, checkpoint_opts.deserialize_chunksize):
#                 args: str = [
#                     (filename, zf.read(filename).decode(encoding='utf-8'), serializer, checkpoint_opts)
#                     for filename in filenames_chunk
#                 ]
#                 payloads_futures: Iterable[DocumentPayload] = executor.map(_process_document_content, args)

#                 for payload in payloads_futures:
#                     yield payload
