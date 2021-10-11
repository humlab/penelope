import os
from typing import Any, Mapping, Optional

from penelope.corpus import DocumentIndexHelper

from .interfaces import InferredModel, TrainingCorpus
from .utility import get_engine_cls_by_method_name

TEMP_PATH = './tmp/'


def train_model(
    train_corpus: TrainingCorpus,
    method: str = 'sklearn_lda',
    engine_args: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> InferredModel:

    os.makedirs(TEMP_PATH, exist_ok=True)

    trained_model = get_engine_cls_by_method_name(method).train(
        train_corpus,
        method,
        engine_args,
        tfidf_weiging=kwargs.get('tfidf_weiging', False),
    )

    train_corpus.document_index = (
        DocumentIndexHelper(train_corpus.document_index).update_counts_by_corpus(train_corpus.corpus).document_index
    )

    return trained_model
