import re
from typing import List, Mapping

import numpy as np


def find_candidate_words(candidate_words: List[str], token2id: Mapping[str, int]):

    word_exprs = [x for x in candidate_words if len(x) > 0 and x.startswith("|") and x.endswith("|")]

    if len(word_exprs) == 0:
        return candidate_words

    words = [w for w in candidate_words if w not in word_exprs]

    for expr in word_exprs:
        pattern = re.compile(expr.strip('|'))  # "^.*tion$"
        words.extend([x for x in token2id if pattern.match(x)])

    return words


def find_n_top_words(word_counts: Mapping[str, int], tokens: List[str], n_top: int) -> List[str]:
    """Returns the `n_top` most frequent word in `tokens`"""
    token_counts = [word_counts.get(w, 0) for w in tokens]
    most_frequent_words = [tokens[x] for x in np.argsort(token_counts)[-n_top:]]
    return most_frequent_words
