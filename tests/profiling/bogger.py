import os
from os.path import join

from penelope import corpus as pc
from penelope import pipeline
from penelope.pipeline.dtm.pipelines import id_tagged_frame_to_DTM_pipeline
from penelope.scripts.utils import load_config

# pylint: disable=unused-argument, too-many-arguments

DATA_PATH: str = '/home/roger/source/welfare-state-analytics/pyriksprot/data/tagged-speech-corpus-id-1965'
# DATA_PATH: str = '/data/riksdagen_corpus_data/tagged-speech-corpus.v0.3.0.id.lemma.no-stopwords.lowercase.feather/'

ARGUMENTS: dict = dict(
    config_filename=join(DATA_PATH, 'corpus.yml'),
    corpus_source=DATA_PATH,
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
)


def debug_main(
    config_filename: str = None,
    corpus_source: str = None,
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
    min_tf: int = None,
):
    config: pipeline.CorpusConfig = load_config(config_filename, corpus_source)

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

    vectorize_opts: pc.VectorizeOpts = pc.VectorizeOpts(already_tokenized=True, min_tf=min_tf, max_tokens=100000)

    corpus_source: str = corpus_source or config.pipeline_payload.source

    corpus: pc.VectorizedCorpus = id_tagged_frame_to_DTM_pipeline(
        corpus_config=config,
        corpus_source=corpus_source,
        file_pattern='**/prot-*.feather',
        extract_opts=extract_opts,
        transform_opts=transform_opts,
        vectorize_opts=vectorize_opts,
    ).value()
    corpus = corpus.slice_by_tf(5)

    os.makedirs('./data/bogger', exist_ok=True)
    corpus.dump(tag='bogger', folder='./data/bogger', mode='files')

    print(f"Stored corpus of shape {corpus.data.shape}")


debug_main(**ARGUMENTS)
