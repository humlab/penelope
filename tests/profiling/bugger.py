import os
from os.path import dirname, isdir, isfile, join

import penelope.corpus as penelope
from penelope import pipeline
from penelope.corpus import TextTransformOpts, remove_hyphens
from penelope.utility import PropertyValueMaskingOpts

# pylint: disable=unused-argument, too-many-arguments

# DATA_PATH: str = join(abspath(join(dirname(__file__), '..')), 'test_data', 'tagged_id_frame_feather')

# DATA_PATH: str = '/data/riksdagen_corpus_data/tagged-speech-corpus.v0.3.0.id.1965'
DATA_PATH: str = '/home/roger/source/welfare-state-analytics/pyriksprot/data/tagged-speech-corpus-id-1965'

ARGUMENTS: dict = dict(
    config_filename=join(DATA_PATH, 'corpus.yml'),
    target_name='profile-topics',
    corpus_source=DATA_PATH,
    train_corpus_folder=None,
    target_folder='./tests/output/',
    fix_hyphenation=False,
    fix_accents=False,
    lemmatize=False,
    pos_includes='',
    pos_excludes='MID|MAD|PAD',
    to_lower=False,
    remove_stopwords=None,
    min_word_length=1,
    max_word_length=None,
    keep_symbols=True,
    keep_numerals=True,
    only_any_alphanumeric=False,
    only_alphabetic=False,
    n_topics=50,
    engine="gensim_lda-multicore",
    passes=None,
    random_seed=None,
    alpha='asymmetric',
    workers=None,
    max_iter=None,
    store_corpus=True,
    store_compressed=True,
    enable_checkpoint=True,
    force_checkpoint=False,
    passthrough_column='lemma',
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
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    passthrough_column: str = None,
):
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(path=config_filename)
    if config.pipeline_payload.source is None:
        config.pipeline_payload.source = corpus_source
        if isdir(corpus_source):
            config.folders(corpus_source, method='replace')
        elif isfile(corpus_source):
            config.folders(dirname(corpus_source), method='replace')

    if passthrough_column is None:

        text_transform_opts: TextTransformOpts = TextTransformOpts()

        text_transform_opts = TextTransformOpts()

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
            text_transform_opts.extra_transforms.append(remove_hyphens)

        transform_opts: penelope.TokensTransformOpts = penelope.TokensTransformOpts(
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

        extract_opts = penelope.ExtractTaggedTokensOpts(
            lemmatize=lemmatize,
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
            **config.pipeline_payload.tagged_columns_names,
        )

    else:
        extract_opts: str = passthrough_column
        text_transform_opts: TextTransformOpts = None
        transform_opts: penelope.TokensTransformOpts = None

    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts()

    engine_args = {
        k: v
        for k, v in {
            'n_topics': n_topics,
            'passes': passes,
            'random_seed': random_seed,
            'alpha': alpha,
            'workers': workers,
            'max_iter': max_iter,
            'work_folder': os.path.join(target_folder, target_name),
        }.items()
        if v is not None
    }

    corpus_source: str = corpus_source or config.pipeline_payload.source

    _: dict = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
            text_transform_opts=text_transform_opts,
        )
        #.filter_tagged_frame()
        .to_dtm(vectorize_opts=None)
        .to_topic_model(
            corpus_source=None,
            train_corpus_folder=train_corpus_folder,
            target_folder=target_folder,
            target_name=target_name,
            engine=engine,
            engine_args=engine_args,
            store_corpus=store_corpus,
            store_compressed=store_compressed,
        )
    ).exhaust()


debug_main(**ARGUMENTS)
