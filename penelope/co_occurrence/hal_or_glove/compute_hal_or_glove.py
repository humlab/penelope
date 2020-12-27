from typing import Iterable, Mapping, Tuple

import more_itertools
import numpy as np
import pandas as pd
import penelope.utility as utility
from penelope.type_alias import FilenameTokensTuple
from tqdm.auto import tqdm

from .vectorizer_glove import GloveVectorizer
from .vectorizer_hal import HyperspaceAnalogueToLanguageVectorizer

logger = utility.getLogger('penelope')

# pylint: disable=
class CoOccurrenceError(ValueError):
    ...


def compute_hal_or_glove_co_occurrences(
    instream: Iterable[Tuple[str, Iterable[str]]],
    *,
    document_index: pd.DataFrame,
    token2id: Mapping[str, int],
    window_size: int,
    distance_metric: int,  # 0, 1, 2
    normalize: str = 'size',
    method: str = 'HAL',
    zero_diagonal: bool = True,
    direction_sensitive: bool = False,
    partition_column: str = 'year',
):
    """Computes co-occurrence as specified by either `Glove` or `Hyperspace Analogous to Hyperspace` (HAL)

        NOTE:
            - Passed document index MUST be in the same sequence as the passed sequence of tokens
    Parameters
    ----------
    corpus : Iterable[str,Iterable[str]]
        Sequence of tokens
    document_index : pd.DataFrame
        Document catalogue
    window_size : int
        [description]
    distance_metric : int
        [description]
    1 : [type]
        [description]
    2normalize : str, optional
        [description], by default 'size'
    method : str, optional
        [description], by default 'HAL'
    zero_diagonal : bool, optional
        [description], by default True
    direction_sensitive : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    # if issubclass(type(corpus), CorpusABC):
    #    doc_terms = [[t.lower().strip('_') for t in terms if len(t) > 2] for terms in corpus.get_texts()]

    if document_index is None:
        raise CoOccurrenceError("expected document index found None")

    if partition_column not in document_index.columns:
        raise CoOccurrenceError(f"expected `{partition_column}` not found in document index")

    if token2id is None:
        raise CoOccurrenceError("expected `token2id` found None")
        # token2id = gensim_utility.build_vocab(doc_terms)

    def get_bucket_key(item: Tuple[str, Iterable[str]]) -> int:

        if not isinstance(item, tuple):
            raise CoOccurrenceError(f"expected stream of (name,tokens) tuples found {type(item)}")

        filename = item[0]
        if not isinstance(str, filename):
            raise CoOccurrenceError(f"expected filename (str) ound {type(filename)}")

        return int(document_index.loc[filename][partition_column])

    total_results = []
    key_streams = more_itertools.bucket(instream, key=get_bucket_key, validator=None)
    keys = sorted(list(key_streams))

    metadata = []
    for i, key in tqdm(enumerate(keys), position=0, leave=True):

        key_stream: FilenameTokensTuple = key_streams[key]
        keyed_document_index = document_index[document_index[partition_column] == key]

        metadata.append(
            dict(
                document_id=i,
                filename='year_{year}.txt',
                document_name='year_{year}',
                year=key,
                n_docs=len(keyed_document_index),
            )
        )

        logger.info(f'Processing{key}...')

        tokens_stream = (tokens for _, tokens in key_stream)

        vectorizer = (
            HyperspaceAnalogueToLanguageVectorizer(token2id=token2id).fit(
                tokens_stream, size=window_size, distance_metric=distance_metric
            )
            if method == "HAL"
            else GloveVectorizer(token2id=token2id).fit(tokens_stream, size=window_size)
        )

        co_occurrence = vectorizer.to_dataframe(
            normalize=normalize, zero_diagonal=zero_diagonal, direction_sensitive=direction_sensitive
        )

        co_occurrence[partition_column] = key

        total_results.append(
            co_occurrence[['year', 'x_term', 'y_term', 'nw_xy', 'nw_x', 'nw_y', 'cwr']],
        )

        # if i == 5: break

    co_occurrences = pd.concat(total_results, ignore_index=True)

    co_occurrences['cwr'] = co_occurrences.cwr / np.max(co_occurrences.cwr, axis=0)

    return co_occurrences
