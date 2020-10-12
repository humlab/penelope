import logging
import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import penelope.corpus.readers as readers
from penelope.corpus import tokenized_corpus, vectorized_corpus

logger = logging.getLogger("corpus_vectorizer")


class CorpusVectorizer:
    def __init__(self, **kwargs):
        self.vectorizer = None
        self.kwargs = kwargs
        self.tokenizer = lambda x: x.split()

    def fit_transform(self, corpus: tokenized_corpus.TokenizedCorpus) -> vectorized_corpus.VectorizedCorpus:

        texts = (' '.join(tokens) for _, tokens in corpus)

        # https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py#L1147
        self.vectorizer = CountVectorizer(tokenizer=self.tokenizer, **self.kwargs)

        bag_term_matrix = self.vectorizer.fit_transform(texts)
        token2id = self.vectorizer.vocabulary_
        document_index = self._document_index(corpus)

        v_corpus = vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, document_index)

        return v_corpus

    def _document_index(self, corpus):
        """Groups matrix by vales in column summing up all values in each category"""
        metadata = corpus.metadata
        df = pd.DataFrame([x.__dict__ for x in metadata], columns=metadata[0].__dict__.keys())
        df['document_id'] = list(df.index)
        return df


def generate_corpus(filename: str, output_folder: str, **kwargs):

    if not os.path.isfile(filename):
        logger.error('no such file: {}'.format(filename))
        return

    dump_tag = '{}_{}_{}_{}'.format(
        os.path.basename(filename).split('.')[0],
        'L{}'.format(kwargs.get('min_len', 0)),
        '-N' if kwargs.get('keep_numerals', False) else '+N',
        '-S' if kwargs.get('keep_symbols', False) else '+S',
    )

    if vectorized_corpus.VectorizedCorpus.dump_exists(dump_tag):
        logger.info('removing existing result files...')
        os.remove(os.path.join(output_folder, '{}_vector_data.npy'.format(dump_tag)))
        os.remove(os.path.join(output_folder, '{}_vectorizer_data.pickle'.format(dump_tag)))

    logger.info('Creating new corpus...')

    reader = readers.TextTokenizer(
        source_path=None,
        filename_pattern=kwargs.get("pattern", "*.txt"),
        tokenize=None,
        as_binary=False,
        fix_whitespaces=True,
        fix_hyphenation=True,
        filename_fields=kwargs.get("filename_fields"),
    )
    corpus = tokenized_corpus.TokenizedCorpus(reader, **kwargs)

    logger.info('Creating document-term matrix...')
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus)

    logger.info('Saving data matrix...')
    v_corpus.dump(tag=dump_tag, folder=output_folder)
