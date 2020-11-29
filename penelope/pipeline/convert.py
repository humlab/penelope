from typing import Iterable

import pandas as pd
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts

from . import interfaces
from .utils import to_text


def to_vectorized_corpus(
    stream: Iterable[interfaces.DocumentPayload], vectorize_opts: VectorizeOpts, document_index: pd.DataFrame
) -> VectorizedCorpus:
    vectorizer = CorpusVectorizer()
    terms = (to_text(payload.content) for payload in stream)
    corpus = vectorizer.fit_transform_(terms, documents=document_index, vectorize_opts=vectorize_opts)
    return corpus
