import multiprocessing as mp
from typing import Iterable, Mapping, Set, Tuple

from loguru import logger
from more_itertools import peekable

from penelope.co_occurrence import ContextOpts, VectorizedTTM, VectorizeType, generate_windows, windows_to_ttm
from penelope.corpus import Token2Id
from penelope.type_alias import DocumentIndex

from ..interfaces import ContentType, DocumentPayload

# logger.add("tokens_to_ttm.log", rotation=None, format="{message}", serialize=False, level="INFO", enqueue=True)

# token2id: dict = None

# def load_token2id(token2id_filename):
#     global token2id
#     if token2id is None:
#         logger.info(f"{mp.current_process().name}: intialize_process_state args={token2id_filename}")
#         with open(token2id_filename, 'rb') as handle:
#             token2id = pickle.load(handle)


def tokens_to_ttm(args) -> dict:
    # global token2id
    try:
        (
            document_id,
            document_name,
            filename,
            token_ids,
            pad_id,
            context_opts,
            concept_ids,
            ignore_ids,
            vocab_size,
        ) = args

        if token_ids and max(token_ids) >= vocab_size:
            raise ValueError("invalid vocab: see issue #159")

        windows: Iterable[Iterable[int]] = generate_windows(
            token_ids=token_ids,
            context_width=context_opts.context_width,
            pad_id=pad_id,
            ignore_pads=context_opts.ignore_padding,
        )

        # windows = list(windows)

        ttm_map: Mapping[VectorizeType, VectorizedTTM] = windows_to_ttm(
            document_id=document_id,
            windows=windows,
            concept_ids=concept_ids,
            ignore_ids=ignore_ids,
            vocab_size=vocab_size,
        )

        return dict(
            document_id=document_id,
            document_name=document_name,
            filename=filename,
            ttm_map=ttm_map,
        )
    except Exception as ex:
        logger.exception(ex)
        raise ex
        # return ex


def prepare_task_stream(
    payload_stream: Iterable[DocumentPayload],
    document_index: DocumentIndex,
    token2id: Token2Id,
    context_opts: ContextOpts,
    concept_ids: Set[int],
    ignore_ids: Set[int],
) -> Iterable[Tuple]:

    fg = token2id.data.get
    # name_to_id: dict = document_index.document_id.to_dict()
    name_to_id: dict = {n: i for n, i in zip(document_index.index, document_index.document_id)}
    task_stream: Iterable[Tuple] = (
        (
            name_to_id[payload.document_name],
            payload.document_name,
            payload.filename,
            payload.content if payload.content_type == ContentType.TOKEN_IDS else [fg(t) for t in payload.content],
            fg(context_opts.pad),
            context_opts,
            concept_ids,
            ignore_ids,
            len(token2id),
        )
        for payload in payload_stream
    )
    return task_stream


def tokens_to_ttm_stream(
    payload_stream: Iterable[DocumentPayload],
    document_index: DocumentIndex,
    token2id: Token2Id,
    context_opts: ContextOpts,
    concept_ids: Set[int],
    ignore_ids: Set[int],
    processes: int = 4,
    chunk_size: int = 25,
) -> Iterable[dict]:

    try:

        args = prepare_task_stream(
            payload_stream=payload_stream,
            document_index=document_index,
            token2id=token2id,
            context_opts=context_opts,
            concept_ids=concept_ids,
            ignore_ids=ignore_ids,
        )
        if not processes:

            for arg in args:
                item: dict = tokens_to_ttm(arg)
                yield item

        else:

            """Force preceeding task to initialize before we spawn processes"""
            args = peekable(args)
            _ = args.peek()

            # token2id_filename: str = f"/tmp/{uuid.uuid4()}.pickle"

            # with open(token2id_filename, 'wb') as handle:
            #     pickle.dump(dict(token2id.data), handle, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Spawning: {processes} processes (chunk size {chunk_size}) ")
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                # , initializer=load_token2id, initargs=(token2id_filename,)) as pool:
                data_futures: Iterable[dict] = pool.imap_unordered(tokens_to_ttm, args, chunksize=chunk_size)
                for item in data_futures:
                    yield item

    except Exception as ex:
        logger.exception(ex)
        raise ex
