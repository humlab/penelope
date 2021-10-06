import os
from typing import Any, Mapping, Optional

from .interfaces import InferredModel, TrainingCorpus
from .utility import add_document_terms_count, get_engine_cls_by_method_name

TEMP_PATH = './tmp/'


def infer_model(
    train_corpus: TrainingCorpus,
    method: str = 'sklearn_lda',
    engine_args: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> InferredModel:

    os.makedirs(TEMP_PATH, exist_ok=True)

    inferred_model = get_engine_cls_by_method_name(method).compute(
        train_corpus,
        method,
        engine_args,
        tfidf_weiging=kwargs.get('tfidf_weiging', False),
    )

    train_corpus.documents = add_document_terms_count(train_corpus.document_index, train_corpus.corpus)

    return inferred_model
