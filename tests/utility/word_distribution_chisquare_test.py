import unittest

import pandas as pd
import penelope.corpus.tokenized_corpus as corpora
import scipy
from penelope.corpus import TokensTransformOpts, vectorizer
from scipy.cluster.hierarchy import linkage  # pylint: disable=unused-import
from tests.utils import create_tokens_reader

unittest.main(argv=['first-arg-is-ignored'], exit=False)


class Test_ChiSquare(unittest.TestCase):
    def setUp(self):
        pass

    def create_reader(self):
        filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_tokens_reader(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
        return reader

    def create_corpus(self):
        reader = self.create_reader()
        tokens_transform_opts = TokensTransformOpts(
            only_any_alphanumeric=True,
            to_lower=True,
            remove_accents=False,
            min_len=2,
            max_len=None,
            keep_numerals=False,
        )
        corpus = corpora.TokenizedCorpus(reader, tokens_transform_opts=tokens_transform_opts)
        return corpus

    def skip_test_chisquare(self):
        corpus = self.create_corpus()
        v = vectorizer.CorpusVectorizer()
        v_corpus = v.fit_transform(corpus, already_tokenized=True).group_by_year().slice_by_n_count(0)
        _ = scipy.stats.chisquare(
            v_corpus.term_bag_matrix.todense(), f_exp=None, ddof=0, axis=0
        )  # pylint: disable=unused-variable
        _ = linkage(v_corpus.term_bag_matrix, 'ward')  # pylint: disable=unused-variable
        results = None
        expected = None
        self.assertEqual(expected, results)


def plot_dists(v_corpus):
    df = pd.DataFrame(v_corpus.bag_term_matrix.toarray(), columns=list(v_corpus.get_feature_names()))
    df['year'] = df.index + 45
    df = df.set_index('year')
    df['year'] = pd.Series(df.index).apply(lambda x: v_corpus.document_index[x][0])
    df[['krig']].plot()  # .loc[df["000"]==49]


# unittest.main(argv=['first-arg-is-ignored'], exit=False)
