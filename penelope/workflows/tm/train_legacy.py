from os.path import join as jj

import penelope.topic_modelling as tm
from penelope.corpus import TextReaderOpts, TextTransformOpts, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import TextTokenizer

# pylint: disable=unused-argument, too-many-arguments


def compute(
    *,
    target_name: str = None,
    corpus_source: str = None,
    target_folder: str = None,
    reader_opts: TextReaderOpts = None,
    text_transform_opts: TextTransformOpts = None,
    transform_opts: TokensTransformOpts = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    n_tokens: int = 200,
    minimum_probability: float = 0.001,
):
    """ runner """

    tokens_reader = TextTokenizer(
        source=corpus_source,
        transform_opts=text_transform_opts,
        reader_opts=reader_opts,
    )

    corpus: TokenizedCorpus = TokenizedCorpus(reader=tokens_reader, transform_opts=transform_opts)

    train_corpus: tm.TrainingCorpus = tm.TrainingCorpus(
        corpus=corpus,
        document_index=corpus.document_index,
        token2id=corpus.token2id,
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

    return dict(folder=target_folder, tag=target_name)
