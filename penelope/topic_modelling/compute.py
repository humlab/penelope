import os
from typing import Any, Mapping, Optional

import penelope.utility as utility

from . import engine_gensim, engine_textacy
from .container import InferredModel, TrainingCorpus
from .utility import add_document_terms_count

logger = utility.getLogger("")

TEMP_PATH = './tmp/'

# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

engines = {'sklearn': engine_textacy, 'gensim_': engine_gensim}


def _find_engine(method: str):
    for key in engines:
        if method.startswith(key):
            return engines[key]
    raise ValueError(f"Unknown method {method}")


def infer_model(
    train_corpus: TrainingCorpus,
    method: str = 'sklearn_lda',
    engine_args: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> InferredModel:

    os.makedirs(TEMP_PATH, exist_ok=True)

    inferred_model = _find_engine(method).compute(
        train_corpus,
        method,
        engine_args,
        tfidf_weiging=kwargs.get('tfidf_weiging', False),
    )

    train_corpus.documents = add_document_terms_count(train_corpus.document_index, train_corpus.corpus)

    return inferred_model
