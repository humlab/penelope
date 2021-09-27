import os

from penelope import topic_modelling
from penelope.corpus import TextReaderOpts, TextTransformOpts, TokenizedCorpus
from penelope.corpus.readers import TextTokenizer


def compute(
    name: str = None,
    corpus_folder: str = None,
    corpus_filename: str = None,
    engine: str = "gensim_lda-multicore",
    topic_modeling_opts: dict = None,
    filename_field: str = None,
    store_corpus: bool = False,
    compressed: bool = True,
):

    if corpus_filename is None and corpus_folder is None:
        raise ValueError("corpus filename")

    if len(filename_field or []) == 0:
        raise ValueError("corpus filename fields")

    if corpus_folder is None:
        corpus_folder, _ = os.path.split(os.path.abspath(corpus_filename))

    target_folder = os.path.join(corpus_folder, name)

    os.makedirs(target_folder, exist_ok=True)

    reader_opts = TextReaderOpts(
        filename_pattern="*.txt",
        filename_filter=None,
        filename_fields=filename_field,
    )

    transform_opts = TextTransformOpts(fix_whitespaces=False, fix_hyphenation=True)

    tokens_reader = TextTokenizer(
        source=corpus_filename,
        transform_opts=transform_opts,
        reader_opts=reader_opts,
        chunk_size=None,
    )

    corpus = TokenizedCorpus(reader=tokens_reader, transform_opts=None)

    train_corpus = topic_modelling.TrainingCorpus(
        terms=corpus.terms,
        doc_term_matrix=None,
        id2word=None,
        document_index=corpus.document_index,
        corpus_options=dict(
            reader_opts=reader_opts.props,
            transform_opts=transform_opts.props,
        ),
    )

    inferred_model = topic_modelling.infer_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=topic_modeling_opts,
    )

    inferred_model.topic_model.save(os.path.join(target_folder, 'gensim.model.gz'))

    topic_modelling.store_model(inferred_model, target_folder, store_corpus=store_corpus, store_compressed=compressed)

    inferred_topics = topic_modelling.compile_inferred_topics_data(
        inferred_model.topic_model, train_corpus.corpus, train_corpus.id2word, train_corpus.document_index
    )

    inferred_topics.store(target_folder)
