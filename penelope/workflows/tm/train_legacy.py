import contextlib
import os

from penelope import corpus as pc
from penelope import pipeline as pp
from penelope import topic_modelling as tm
from penelope.corpus.readers import TextTokenizer
from penelope.topic_modelling.engines.engine_gensim.options import SUPPORTED_ENGINES

# pylint: disable=unused-argument, too-many-arguments

jj = os.path.join


def compute(
    *,
    target_name: str = None,
    corpus_source: str = None,
    target_folder: str = None,
    reader_opts: pc.TextReaderOpts = None,
    text_transform_opts: pc.TextTransformOpts = None,
    transform_opts: pc.TokensTransformOpts = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    n_tokens: int = 200,
    minimum_probability: float = 0.001,
):

    tokens_reader: TextTokenizer = TextTokenizer(
        source=corpus_source,
        transform_opts=text_transform_opts,
        reader_opts=reader_opts,
    )

    corpus: pc.TokenizedCorpus = pc.TokenizedCorpus(reader=tokens_reader, transform_opts=transform_opts)

    train_corpus: tm.TrainingCorpus = tm.TrainingCorpus(
        corpus=corpus,
        corpus_options=dict(
            reader_opts=reader_opts.props,
            transform_opts=text_transform_opts.props,
        ),
    )

    inferred_model: tm.InferredModel = tm.train_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=engine_args,
    )

    inferred_model.topic_model.save(jj(target_folder, 'gensim.model.gz'))

    inferred_model.store(target_folder, store_compressed=store_compressed)

    if store_corpus:
        train_corpus.store(target_folder)

    inferred_topics: tm.InferredTopicsData = tm.predict_topics(
        inferred_model.topic_model,
        corpus=train_corpus.corpus,
        id2token=train_corpus.id2token,
        document_index=train_corpus.document_index,
        n_tokens=n_tokens,
        minimum_probability=minimum_probability,
    )

    inferred_topics.store(target_folder)

    """Store a reconstructed corpus config in target folder"""
    with contextlib.suppress(Exception):
        pp.CorpusConfig.create(
            corpus_name=target_name,
            corpus_type=pp.CorpusType.Text,
            corpus_pattern=None,
            checkpoint_opts=None,
            text_reader_opts=reader_opts,
            text_transform_opts=text_transform_opts,
            pipelines=None,
            pipeline_payload=pp.PipelinePayload(source=corpus_source),
            language=transform_opts.language,
        ).dump(jj(target_folder, "corpus.yml"))

    return dict(folder=target_folder, tag=target_name)


def compute2(
    *,
    target_name: str = None,
    corpus_source: str = None,
    corpus_folder: str = None,
    filename_field: str = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    n_tokens: int = 200,
    minimum_probability: float = 0.001,
):

    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"Engine {engine} not supported or deprecated")

    if corpus_source is None and corpus_folder is None:
        raise ValueError("corpus filename")

    if len(filename_field or []) == 0:
        raise ValueError("corpus filename fields")

    if corpus_folder is None:
        corpus_folder, _ = os.path.split(os.path.abspath(corpus_source))

    target_folder: str = os.path.join(corpus_folder, target_name)

    os.makedirs(target_folder, exist_ok=True)

    reader_opts: pc.TextReaderOpts = pc.TextReaderOpts(
        filename_pattern="*.txt",
        filename_filter=None,
        filename_fields=filename_field,
    )

    text_transform_opts: pc.TextTransformOpts = pc.TextTransformOpts(fix_whitespaces=False, fix_hyphenation=True)
    transform_opts: pc.TokensTransformOpts = pc.TokensTransformOpts()

    return compute(
        target_name=target_name,
        corpus_source=corpus_source,
        target_folder=target_folder,
        reader_opts=reader_opts,
        text_transform_opts=text_transform_opts,
        transform_opts=transform_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
        n_tokens=n_tokens,
        minimum_probability=minimum_probability,
    )
