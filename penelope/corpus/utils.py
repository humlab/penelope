from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import scipy.sparse as sp
from loguru import logger
from tqdm.auto import tqdm

from penelope.utility import flatten


def generate_token2id(terms: Iterator[Iterator[str]], n_docs: int = None) -> dict:

    token2id = defaultdict()
    token2id.default_factory = token2id.__len__
    tokens_iter = tqdm(
        terms, desc="Vocab", total=n_docs, position=0, mininterval=1.0, leave=True
    )  # if n_docs > 0 else terms
    for tokens in tokens_iter:
        for token in tokens:
            _ = token2id[token]
    return dict(token2id)


def bow2text(document: List[Tuple[int, int]], id2token: Dict[int, str]) -> str:
    """Creates a text corpus out of a BoW corpus, repeating words in sequence."""
    return ' '.join(flatten([f * [id2token[token_id]] for token_id, f in document]))


def csr2bow(csr: sp.spmatrix) -> Iterable[List[Tuple[int, float]]]:
    """
    CSR is the standard CSR representation where the column indices for row i are stored in indices
    [indptr[i]:indptr[i+1]] and their corresponding values are stored in data[indptr[i]:indptr[i+1]].

    data is an array containing all the non zero elements of the sparse matrix.
    indices is an array mapping each element in data to its column in the sparse matrix.
    indptr then maps the elements of data and indices to the rows of the sparse matrix. This is done with the following reasoning:

    """
    assert sp.issparse(csr)

    bow: Iterable[Tuple[int, float]] = None

    if not sp.isspmatrix_csr(csr):
        logger.warning("csr2bow: called with non csr (inefficient")
        csr = csr.tocsr()

    if sp.isspmatrix_csr(csr):

        data = csr.data
        indices = csr.indices
        indptr = csr.indptr

        bow: Iterable[Tuple[int, float]] = (
            list(zip(indices[indptr[i] : indptr[i + 1]], data[indptr[i] : indptr[i + 1]]))
            for i in range(0, csr.shape[0])
        )

    return bow


def term_frequency(tokens: Sequence[str], counts: dict = None) -> Mapping[str, int]:
    counts = counts or defaultdict(int)
    for v in tokens:
        counts[v] += 1
    return counts
