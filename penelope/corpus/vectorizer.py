import logging
import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from penelope.corpus.tokenized_corpus import TokenizedCorpus

from . import readers, tokenized_corpus, vectorized_corpus

logger = logging.getLogger("corpus_vectorizer")


class CorpusVectorizer:
    def __init__(self, **kwargs):
        self.vectorizer = None
        self.kwargs = kwargs
        self.tokenizer = lambda x: x.split()

    # FIXME Allow for non-tokenized corpus to be passed in
    def fit_transform(self, corpus: tokenized_corpus.TokenizedCorpus) -> vectorized_corpus.VectorizedCorpus:

        # if isinstance(corpus, TokenizedCorpus):
        texts = (' '.join(tokens) for _, tokens in corpus)
        # elif isinstance()
        self.vectorizer = CountVectorizer(tokenizer=self.tokenizer, **self.kwargs)

        bag_term_matrix = self.vectorizer.fit_transform(texts)
        token2id = self.vectorizer.vocabulary_
        documents = corpus.documents

        v_corpus = vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, documents)

        return v_corpus


def generate_corpus(filename: str, output_folder: str, **kwargs):
    """[summary]

    Parameters
    ----------
    filename : str
        Source filename
    output_folder : str
        Target folder
    """
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
