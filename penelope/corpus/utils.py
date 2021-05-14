from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

from penelope.utility import flatten
from tqdm import tqdm


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


def bow_to_text(document: List[Tuple[int, int]], id2token: Dict[int, str]) -> str:
    return ' '.join(flatten([f * [id2token[token_id]] for token_id, f in document]))
