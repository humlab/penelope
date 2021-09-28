from penelope import corpus as corpora
from penelope.notebook import word_trends


def test_isolate_bug():

    corpus_folder: str = '/data/westac/data/APA'
    corpus_tag: str = 'APA'
    corpus: corpora.VectorizedCorpus = corpora.VectorizedCorpus.load(folder=corpus_folder, tag=corpus_tag)

    opts = {'corpus': corpus, 'corpus_folder': corpus_folder, 'corpus_tag': corpus_tag, 'n_count': 25000}

    trends_data: word_trends.TrendsData = word_trends.TrendsData(**opts)

    assert trends_data is not None
