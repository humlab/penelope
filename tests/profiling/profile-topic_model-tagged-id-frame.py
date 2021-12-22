from os.path import join

from penelope.scripts import topic_model

# pylint: disable=unused-argument, too-many-arguments


def debug_main():

    # data_path: str = join(abspath(join(dirname(__file__), '..')), 'test_data', 'tagged_id_frame_feather')

    # data_path: str = '/data/riksdagen_corpus_data/tagged-speech-corpus.v0.3.0.id.1965'
    data_path: str = '/home/roger/source/welfare-state-analytics/pyriksprot/data/tagged-speech-corpus-id-1965'

    arguments: dict = dict(
        config_filename=join(data_path, 'corpus.yml'),
        target_name='profile-topics',
        corpus_source=data_path,
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
    topic_model._main(**arguments)  # pylint: disable=protected-access


debug_main()
