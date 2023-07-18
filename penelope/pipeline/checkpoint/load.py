import zipfile
from multiprocessing import get_context
from typing import Iterable, Protocol, Union

from loguru import logger
from tqdm import tqdm

from penelope.corpus.serialize import IContentSerializer, LoaderRegistry, SerializeOpts, SerializerRegistry

from ..interfaces import DocumentPayload


class PayloadLoader(Protocol):
    def __call__(
        self,
        zip_or_filename: str,
        opts: SerializeOpts,
        filenames: list[str],
        ordered: bool = False,
    ) -> Iterable[DocumentPayload]:
        ...


def load_payload(
    zip_or_filename: str, filename: str, opts: SerializeOpts, serializer: IContentSerializer
) -> DocumentPayload:
    payload: DocumentPayload = DocumentPayload(
        content_type=opts.content_type,
        content=LoaderRegistry.get_loader(opts)(
            zip_or_filename=zip_or_filename,
            filename=filename,
            opts=opts,
            serializer=serializer,
        ),
        filename=filename,
    )
    tfs: dict = serializer.compute_term_frequency(content=payload.content, options=opts)
    if tfs:
        payload.remember(**tfs)
    return payload


def load_payloads_singleprocess(
    *, zip_or_filename: Union[str, zipfile.ZipFile], opts: SerializeOpts, filenames: list[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    logger.debug("Using sequential deserialization")
    if opts.feather_folder:
        logger.debug(f"Using feather checkpoint folder {opts.feather_folder}.")

    serializer: IContentSerializer = SerializerRegistry.create(opts)
    return (load_payload(zip_or_filename, filename, opts, serializer) for filename in filenames)


def _multiprocess_load_task(args: tuple) -> DocumentPayload:
    filename, zip_or_filename, serializer, opts = args
    try:
        payload = load_payload(zip_or_filename, filename, opts, serializer)
        return payload
    except Exception as ex:
        logger.error(f"Filename: {filename}")
        logger.exception(ex)
        raise ex
        # return ex


def load_payloads_multiprocess(
    *,
    zip_or_filename: str,
    opts: SerializeOpts,
    filenames: list[str],
    ordered: bool = False,
) -> Iterable[DocumentPayload]:
    try:
        logger.trace(f"Using parallel deserialization with {opts.deserialize_processes} processes.")
        if opts.feather_folder:
            logger.trace(f"Using feather checkpoint folder {opts.feather_folder}.")

        serializer: IContentSerializer = SerializerRegistry.create(opts)

        args: Iterable[tuple] = tqdm(
            [(filename, zip_or_filename, serializer, opts) for filename in filenames], desc="read"
        )

        with get_context("spawn").Pool(processes=opts.deserialize_processes) as executor:
            mapper = executor.imap_unordered if not ordered else executor.imap
            payloads_futures: Iterable[DocumentPayload] = mapper(
                _multiprocess_load_task, args, chunksize=opts.deserialize_chunksize
            )

            for payload in payloads_futures:
                yield payload

    except Exception as ex:
        logger.exception(ex)
        raise ex
        # return ex
