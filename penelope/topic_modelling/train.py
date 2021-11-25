import os
from typing import Any, Mapping, Optional

from .engines import get_engine_cls_by_method_name
from .interfaces import InferredModel, TrainingCorpus


def train_model(
    train_corpus: TrainingCorpus,
    method: str = 'sklearn_lda',
    engine_args: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> InferredModel:

    if engine_args.get('work_folder', False):
        os.makedirs(engine_args.get('work_folder'), exist_ok=True)

    trained_model = get_engine_cls_by_method_name(method).train(
        train_corpus,
        method,
        engine_args,
        tfidf_weiging=kwargs.get('tfidf_weiging', False),
    )

    train_corpus.update_word_counts()

    return trained_model
