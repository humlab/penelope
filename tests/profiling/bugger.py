import os
from os.path import join

from penelope import corpus as pc
from penelope import pipeline
from penelope.corpus.dtm.vectorizer import VectorizeOpts
from penelope.pipeline.topic_model.pipelines import from_id_tagged_frame_pipeline
from penelope.scripts.utils import load_config, remove_none

# pylint: disable=unused-argument, too-many-arguments

# DATA_PATH: str = join(abspath(join(dirname(__file__), '..')), 'test_data', 'tagged_id_frame_feather')

# DATA_PATH: str = '/data/riksdagen_corpus_data/tagged-speech-corpus.v0.3.0.id.1965'
DATA_PATH: str = '/home/roger/source/welfare-state-analytics/pyriksprot/data/tagged-speech-corpus-id-1965'
# DATA_PATH: str = '/home/roger/source/penelope/tests/test_data/tagged_id_frame_feather'

ARGUMENTS: dict = dict(
    config_filename=join(DATA_PATH, 'corpus.yml'),
    target_name='profile-topics',
    corpus_source=DATA_PATH,
    train_corpus_folder=None,
    target_folder='./tests/output/',
    fix_hyphenation=False,
    fix_accents=False,
    lemmatize=True,
    pos_includes='',
    pos_excludes='',
    to_lower=False,
    remove_stopwords=None,
    min_word_length=1,
    max_word_length=None,
    keep_symbols=True,
    keep_numerals=True,
    only_any_alphanumeric=False,
    only_alphabetic=False,
    max_tokens=500000,
    alpha='asymmetric',
    chunk_size=2000,
    engine="gensim_lda-multicore",
    max_iter=4000,
    n_topics=50,
    passes=1,
    random_seed=None,
    minimum_probability=0.01,
    workers=3,
    store_corpus=True,
    store_compressed=True,
    # passthrough_column='lemma',
)


def debug_main(
    config_filename: str = None,
    target_name: str = None,
    corpus_source: str = None,
    train_corpus_folder: str = None,
    target_folder: str = None,
    fix_hyphenation: bool = True,
    fix_accents: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    max_tokens: int = None,
    alpha: str = 'asymmetric',
    chunk_size: int = 2000,
    engine: str = "gensim_lda-multicore",
    max_iter: int = None,
    minimum_probability: float = None,
    n_topics: int = 50,
    passes: int = None,
    per_word_topics: bool = False,
    random_seed: int = None,
    update_every: int = 1,
    workers: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    passthrough_column: str = None,
):
    config: pipeline.CorpusConfig = load_config(config_filename, corpus_source)

    if passthrough_column is None:

        text_transform_opts: pc.TextTransformOpts = pc.TextTransformOpts()

        if fix_accents:
            text_transform_opts.fix_accents = True

        if fix_hyphenation:
            """Replace default dehyphen function"""
            # fix_hyphens: Callable[[str], str] = (
            #     remove_hyphens_fx(config.text_reader_opts.dehyphen_expr)
            #     if config.text_reader_opts.dehyphen_expr is not None
            #     else remove_hyphens
            # )
            text_transform_opts.fix_hyphenation = False
            text_transform_opts.extra_transforms.append(pc.remove_hyphens)

        transform_opts: pc.TokensTransformOpts = pc.TokensTransformOpts(
            to_lower=to_lower,
            to_upper=False,
            min_len=min_word_length,
            max_len=max_word_length,
            remove_accents=False,
            remove_stopwords=(remove_stopwords is not None),
            stopwords=None,
            extra_stopwords=None,
            language=remove_stopwords,
            keep_numerals=keep_numerals,
            keep_symbols=keep_symbols,
            only_alphabetic=only_alphabetic,
            only_any_alphanumeric=only_any_alphanumeric,
        )

        extract_opts = pc.ExtractTaggedTokensOpts(
            lemmatize=lemmatize,
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
        ).set_numeric_names()

    else:
        # extract_opts: str = passthrough_column
        text_transform_opts: pc.TextTransformOpts = None
        # transform_opts: penelope.TokensTransformOpts = None

    engine_args = remove_none(
        {
            'n_topics': n_topics,
            'passes': passes,
            'random_seed': random_seed,
            'alpha': alpha,
            'workers': workers,
            'max_iter': max_iter,
            'work_folder': os.path.join(target_folder, target_name),
            'chunk_size': chunk_size,
            'update_every': 2,
        }
    )
    vectorize_opts: VectorizeOpts = VectorizeOpts(
        already_tokenized=True,
        max_tokens=max_tokens,
        lowercase=False,
    )
    corpus_source: str = corpus_source or config.pipeline_payload.source

    _: dict = from_id_tagged_frame_pipeline(
        corpus_config=config,
        corpus_source=corpus_source,
        file_pattern='**/*.feather',
        extract_opts=extract_opts,
        transform_opts=transform_opts,
        vectorize_opts=vectorize_opts,
        target_name=target_name,
        train_corpus_folder=train_corpus_folder,
        target_folder=target_folder,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
    ).value()


debug_main(**ARGUMENTS)
