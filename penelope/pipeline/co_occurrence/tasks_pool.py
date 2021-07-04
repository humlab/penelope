from multiprocessing import get_context
from typing import Iterable, Mapping, Tuple

from tqdm import tqdm

from penelope.co_occurrence import VectorizedTTM, VectorizeType, windows_to_ttm
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus.dtm import WORD_PAIR_DELIMITER

sj = WORD_PAIR_DELIMITER.join


def tokens_to_ttm(args) -> dict:
    document_id, document_name, token_ids, pad_id, context_opts, concept_ids, ignore_ids, vocab_size = args
    windows: Iterable[Iterable[int]] = generate_windows(
        token_ids=token_ids,
        context_width=context_opts.context_width,
        pad_id=pad_id,
    )

    windows = list(windows)

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
        ttm_map=ttm_map,
    )


def tokens_to_ttm_stream(args: Iterable[Tuple]) -> Iterable[dict]:

    with get_context("spawn").Pool(processes=6) as executor:
        data_futures: Iterable[dict] = executor.map(tokens_to_ttm, args, chunksize=10)
        for item in tqdm(data_futures):
            yield item
