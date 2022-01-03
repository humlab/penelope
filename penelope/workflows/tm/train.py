import os

from penelope import topic_modelling as tm
from penelope.corpus import TextReaderOpts, TextTransformOpts, TokenizedCorpus
from penelope.corpus.readers import TextTokenizer
from penelope.topic_modelling.engines.engine_gensim.options import SUPPORTED_ENGINES

# pylint: disable=too-many-arguments

# FIXME: Add target_mode/trained_model_folder? Or leave be as legacy...


def compute(
    name: str = None,
    corpus_folder: str = None,
    corpus_source: str = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    filename_field: str = None,
    minimum_probability: float = 0.001,
    n_tokens: int = 200,
    store_corpus: bool = False,
    compressed: bool = True,
):

    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"Engine {engine} not supported or deprecated")

    if corpus_source is None and corpus_folder is None:
        raise ValueError("corpus filename")

    if len(filename_field or []) == 0:
        raise ValueError("corpus filename fields")

    if corpus_folder is None:
        corpus_folder, _ = os.path.split(os.path.abspath(corpus_source))

    target_folder = os.path.join(corpus_folder, name)

    os.makedirs(target_folder, exist_ok=True)

    reader_opts = TextReaderOpts(
        filename_pattern="*.txt",
        filename_filter=None,
        filename_fields=filename_field,
    )

    transform_opts = TextTransformOpts(fix_whitespaces=False, fix_hyphenation=True)

    tokens_reader = TextTokenizer(
        source=corpus_source,
        transform_opts=transform_opts,
        reader_opts=reader_opts,
    )

    corpus: TokenizedCorpus = TokenizedCorpus(reader=tokens_reader, transform_opts=None)

    train_corpus: tm.TrainingCorpus = tm.TrainingCorpus(
        corpus=corpus,
        corpus_options=dict(
            reader_opts=reader_opts.props,
            transform_opts=transform_opts.props,
        ),
    )

    inferred_model: tm.InferredModel = tm.train_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=engine_args,
    )

    inferred_model.topic_model.save(os.path.join(target_folder, 'gensim.model.gz'))

    inferred_model.store(target_folder, store_compressed=compressed)

    if store_corpus:
        train_corpus.store(target_folder)

    inferred_topics: tm.InferredTopicsData = tm.predict_topics(
        inferred_model.topic_model,
        corpus=train_corpus.corpus,
        id2token=train_corpus.id2token,
        document_index=train_corpus.document_index,
        minimum_probability=minimum_probability,
        n_tokens=n_tokens,
    )

    inferred_topics.store(target_folder)
