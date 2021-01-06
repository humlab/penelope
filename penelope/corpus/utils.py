from collections import defaultdict
from typing import Iterator

from tqdm import tqdm


def generate_token2id(terms: Iterator[Iterator[str]], n_docs=None):

    token2id = defaultdict()
    token2id.default_factory = token2id.__len__
    tokens_iter = tqdm(terms, desc="Vocab", total=n_docs, position=0, leave=True) if n_docs > 0 else terms
    for tokens in tokens_iter:
        for token in tokens:
            _ = token2id[token]
        tokens_iter.set_description(f"Vocab #{len(token2id)}")
    return dict(token2id)
