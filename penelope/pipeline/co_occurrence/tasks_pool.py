from multiprocessing import get_context
from typing import Iterable, Mapping, Tuple

from loguru import logger
from penelope.co_occurrence import VectorizedTTM, VectorizeType, windows_to_ttm
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus.dtm import WORD_PAIR_DELIMITER

sj = WORD_PAIR_DELIMITER.join

logger.add("tokens_to_ttm.log", rotation=None, format="{message}", serialize=False, level="INFO", enqueue=True)


def tokens_to_ttm(args) -> dict:
    try:
        document_id, document_name, filename, token_ids, pad_id, context_opts, concept_ids, ignore_ids, vocab_size = args
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


def tokens_to_ttm_stream(args: Iterable[Tuple], processes: int = 4, chunksize: int = 25) -> Iterable[dict]:

    try:

        if processes is None:
            for arg in args:
                item: dict = tokens_to_ttm(arg)
                yield item
        else:
            logger.info(f"Spawning: {processes} processes (chunksize {chunksize}) ")
            with get_context("spawn").Pool(processes=processes) as pool:
                data_futures: Iterable[dict] = pool.imap(tokens_to_ttm, args, chunksize=chunksize)
                for item in data_futures:
                    yield item
    except Exception as ex:
        logger.exception(ex)
        raise ex
