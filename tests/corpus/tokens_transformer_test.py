from penelope.corpus import TokensTransformer, TokensTransformOpts
from penelope.corpus.transforms import load_stopwords
from penelope.vendor.nltk import STOPWORDS_CACHE, extended_stopwords


def test_transform_smoke_test():
    transformer = TokensTransformer(tokens_transform_opts=TokensTransformOpts())

    assert transformer is not None


def test_transform_load_stopwords():

    stopwords = load_stopwords("swedish")

    assert isinstance(stopwords, set)

    assert 'och' in stopwords

    stopwords_plus = load_stopwords("swedish", {"apa", "paj"})

    assert stopwords_plus == stopwords + {"apa", "paj"}


def test_remove_stopwords():

    stopwords = load_stopwords("swedish")

    xtra_stopwords = extended_stopwords("swedish")

    assert xtra_stopwords > stopwords
    assert len(STOPWORDS_CACHE) > 0
