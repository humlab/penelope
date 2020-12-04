from typing import Iterable, List

import pandas as pd
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts, default_tokenizer

from . import interfaces


def _payload_tokens(payload: interfaces.DocumentPayload) -> List[str]:
    if payload.previous_content_type == interfaces.ContentType.TEXT:
        return (payload.content[0], default_tokenizer(payload.content[1]))
    return payload.content


def to_vectorized_corpus(
    stream: Iterable[interfaces.DocumentPayload], vectorize_opts: VectorizeOpts, document_index: pd.DataFrame
) -> VectorizedCorpus:
    vectorizer = CorpusVectorizer()
    vectorize_opts.already_tokenized = True
    terms = (_payload_tokens(payload) for payload in stream)
    corpus = vectorizer.fit_transform_(terms, document_index=document_index, vectorize_opts=vectorize_opts)
    return corpus
