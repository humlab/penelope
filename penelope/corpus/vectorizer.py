import logging
import os
from typing import Callable, Iterable, Mapping, Tuple, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from . import readers, tokenized_corpus, vectorized_corpus

logger = logging.getLogger("corpus_vectorizer")


def _default_tokenizer(lowercase=True):
    def _lowerccase_tokenize(tokens):
        return [x.lower() for x in tokens]

    def _no_tokenize(tokens):
        return tokens

    if lowercase:
        return lambda tokens: [x.lower() for x in tokens]

    return _lowerccase_tokenize if lowercase else _no_tokenize


def _no_tokenize(tokens):
    return tokens


DocumentTermsStream = Iterable[Tuple[str, Iterable[str]]]


class CorpusVectorizer:
    def __init__(self):
        self.vectorizer = None
        self.vectorizer_opts = {}

    def fit_transform(
        self,
        corpus: Union[tokenized_corpus.TokenizedCorpus, DocumentTermsStream],
        *,
        vocabulary: Mapping[str, int] = None,
        tokenizer: Callable = None,
        lowercase: bool = False,
        stop_words: str = None,
        max_df: float = 1.0,
        min_df: int = 1,
    ) -> vectorized_corpus.VectorizedCorpus:
        """Returns a vectorized corpus from of `corpus`

        Note:
          -  Input texts are already tokenized, so tokenizer is an identity function

        Parameters
        ----------
        corpus : tokenized_corpus.TokenizedCorpus
            [description]

        Returns
        -------
        vectorized_corpus.VectorizedCorpus
            [description]
        """

        if vocabulary is None:
            if hasattr(corpus, 'vocabulary'):
                vocabulary = corpus.vocabulary
            elif hasattr(corpus, 'token2id'):
                vocabulary = corpus.token2id

        if tokenizer is None:  # Iterator[Tuple[str,Iterator[str]]]
            tokenizer = _no_tokenize
            if lowercase:
                tokenizer = lambda tokens: [t.lower() for t in tokens]
            lowercase = False

        vectorizer_opts = dict(
            tokenizer=tokenizer,
            lowercase=lowercase,
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            vocabulary=vocabulary,
        )

        if hasattr(corpus, 'terms'):
            terms = corpus.terms
        else:
            terms = (x[1] for x in corpus)

        self.vectorizer = CountVectorizer(**vectorizer_opts)
        self.vectorizer_opts = vectorizer_opts

        bag_term_matrix = self.vectorizer.fit_transform(terms)
        token2id = self.vectorizer.vocabulary_

        if hasattr(corpus, 'documents'):
            documents = corpus.documents
        else:
            logger.warning("corpus has no `documents` property (generating a dummy index")
            documents = pd.DataFrame(
                data=[{'index': i, 'filename': f'file_{i}.txt'} for i in range(0, bag_term_matrix.shape[0])]
            ).set_index('index')
            documents['document_id'] = documents.index
        # ignored_words = self.vectorizer.stop_words_

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
        source=None,
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
