from typing import Iterable

import pandas as pd
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts
from penelope.utility import to_text

from . import interfaces


def to_vectorized_corpus(
    stream: Iterable[interfaces.DocumentPayload], vectorize_opts: VectorizeOpts, document_index: pd.DataFrame
) -> VectorizedCorpus:
    vectorizer = CorpusVectorizer()
    terms = ((payload.content[0], to_text(payload.content[1])) for payload in stream)
    corpus = vectorizer.fit_transform_(terms, document_index=document_index, vectorize_opts=vectorize_opts)
    return corpus
