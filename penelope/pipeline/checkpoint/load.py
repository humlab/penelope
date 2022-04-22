import os
import zipfile
from multiprocessing import get_context
from typing import Callable, Iterable, List, Tuple, Union

import pandas as pd
from loguru import logger
from tqdm import tqdm

from penelope.type_alias import TaggedFrame
from penelope.utility.zip_utils import zipfile_or_filename

from ..interfaces import ContentType, DocumentPayload
from .interface import CheckpointOpts, IContentSerializer, Serializer, TaggedFrameStore
from .serialize import create_serializer

PayloadLoader = Callable[[str, CheckpointOpts, List[str], bool], Iterable[DocumentPayload]]


@zipfile_or_filename(mode='r')
def load_tagged_frame(
    *, zip_or_filename: TaggedFrameStore, filename: str, checkpoint_opts: CheckpointOpts, serializer: Serializer
) -> TaggedFrame:
    tagged_frame: TaggedFrame = serializer.deserialize(
        content=zip_or_filename.read(filename).decode(encoding='utf-8'),
        options=checkpoint_opts,
    )
    if checkpoint_opts.lower_lemma:
        tagged_frame[checkpoint_opts.lemma_column] = pd.Series(
            [x.lower() for x in tagged_frame[checkpoint_opts.lemma_column]], dtype=object
        )
    return tagged_frame


def load_feathered_tagged_frame(
    *, zip_or_filename: TaggedFrameStore, filename: str, checkpoint_opts: CheckpointOpts, serializer: Serializer
) -> pd.DataFrame:
    feather_filename: str = checkpoint_opts.feather_filename(filename)
    if os.path.isfile(feather_filename):
        tagged_frame: pd.DataFrame = pd.read_feather(feather_filename)
        if checkpoint_opts.lower_lemma:
            if len(tagged_frame) > 0:
                tagged_frame[checkpoint_opts.lemma_column] = tagged_frame[checkpoint_opts.lemma_column].str.lower()
    else:
        tagged_frame = load_tagged_frame(
            zip_or_filename=zip_or_filename,
            filename=filename,
            checkpoint_opts=checkpoint_opts,
            serializer=serializer,
        )
        tagged_frame.reset_index(drop=True).to_feather(feather_filename, compression="lz4")
    return tagged_frame


def get_checkpoint_loader(checkpoint_opts: CheckpointOpts) -> Callable:

    if checkpoint_opts.content_type == ContentType.TAGGED_FRAME:
        if checkpoint_opts.feather_folder:
            return load_feathered_tagged_frame
        return load_tagged_frame

    raise NotImplementedError("Loader so far only implemented for TAGGED FRAME")


def load_payload(
    zip_or_filename: str, filename: str, checkpoint_opts: CheckpointOpts, serializer: IContentSerializer
) -> DocumentPayload:

    payload: DocumentPayload = DocumentPayload(
        content_type=checkpoint_opts.content_type,
        content=get_checkpoint_loader(checkpoint_opts)(
            zip_or_filename=zip_or_filename,
            filename=filename,
            checkpoint_opts=checkpoint_opts,
            serializer=serializer,
        ),
        filename=filename,
    )
    tfs: dict = serializer.compute_term_frequency(content=payload.content, options=checkpoint_opts)
    if tfs:
        payload.remember(**tfs)
    return payload


def load_payloads_singleprocess(
    *, zip_or_filename: Union[str, zipfile.ZipFile], checkpoint_opts: CheckpointOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    logger.debug("Using sequential deserialization")
    if checkpoint_opts.feather_folder:
        logger.debug(f"Using feather checkpoint folder {checkpoint_opts.feather_folder}.")

    serializer: IContentSerializer = create_serializer(checkpoint_opts)
    return (load_payload(zip_or_filename, filename, checkpoint_opts, serializer) for filename in filenames)


def _multiprocess_load_task(args: Tuple) -> DocumentPayload:
    filename, zip_or_filename, serializer, checkpoint_opts = args
    try:
        payload = load_payload(zip_or_filename, filename, checkpoint_opts, serializer)
        return payload
    except Exception as ex:
        logger.error(f"Filename: {filename}")
        logger.exception(ex)
        raise ex
        # return ex


def load_payloads_multiprocess(
    *,
    zip_or_filename: str,
    checkpoint_opts: CheckpointOpts,
    filenames: List[str],
    ordered: bool = False,
) -> Iterable[DocumentPayload]:

    try:

        logger.trace(f"Using parallel deserialization with {checkpoint_opts.deserialize_processes} processes.")
        if checkpoint_opts.feather_folder:
            logger.trace(f"Using feather checkpoint folder {checkpoint_opts.feather_folder}.")

        serializer: IContentSerializer = create_serializer(checkpoint_opts)

        args: Iterable[Tuple] = tqdm(
            [(filename, zip_or_filename, serializer, checkpoint_opts) for filename in filenames], desc="read"
        )

        with get_context("spawn").Pool(processes=checkpoint_opts.deserialize_processes) as executor:

            mapper = executor.imap_unordered if not ordered else executor.imap
            payloads_futures: Iterable[DocumentPayload] = mapper(
                _multiprocess_load_task, args, chunksize=checkpoint_opts.deserialize_chunksize
            )

            for payload in payloads_futures:
                yield payload

    except Exception as ex:
        logger.exception(ex)
        raise ex
        # return ex
