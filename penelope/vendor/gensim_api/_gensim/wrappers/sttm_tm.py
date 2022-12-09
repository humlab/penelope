# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code is based on gensim.models.wrapper.LdaMallet
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
r"""Python wrapper for `STTM Short text topic modelling library <https://github.com/qiang2100/STTM>`_

Notes
-----

Installation
------------

Examples
--------

"""
import logging
import os
import random
import tempfile
from itertools import chain
from typing import Iterable

import numpy
import scipy
from smart_open import smart_open

try:
    from gensim import matutils, utils
    from gensim.models import basemodel
    from gensim.utils import check_output
except ImportError:
    ...

logger = logging.getLogger(__name__)

AVALIABLE_MODELS = "LDA BTM PTM SATM DMM WATM".split()

# pylint: disable=too-many-instance-attributes, too-many-arguments


class STTMTopicModel(utils.SaveLoad, basemodel.BaseTopicModel):
    """Python wrapper for SSTM using `SSTM <https://github.com/qiang2100/STTM>`_."""

    def __init__(
        self,
        sstm_jar_path,
        model,
        corpus,
        id2word=None,
        vectors=None,
        num_topics=20,
        alpha=0.1,
        beta=0.01,
        iterations=2000,
        prefix='results/',
        name='model',
        twords=20,
        sstep=0,
    ):
        """

        Parameters
        ----------
        sstm_path : str
            Path to the SSTM jar file.
        corpus : iterable of iterable of (int, int), optional
            Collection of texts in BoW format.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping between tokens ids and words from corpus, if not specified - will be inferred from `corpus`.
        vectors:
            Path to the word2vec file.
        num_topics : int, optional
            Number of topics.
        alpha : int, optional
            Alpha hyperparameter.
        beta : int, optional
            Beta hyperparameter.
        iterations : int, optional
            Number of training iterations.
        prefix : str, optional
            Prefix for produced temporary files.
        name : str, optional
            Name of topic model experiment.
        twords: int, optional
            Number of the most probable topical words.
        sstep : int, optional
            Step to save the sampling outputs.

        """
        self.avaliable_models = AVALIABLE_MODELS
        self.sstm_jar_path: str = sstm_jar_path
        self.model = model.upper()
        if self.model not in self.avaliable_models:
            raise ValueError("unknown model")

        self.id2word = id2word
        self.vectors = vectors
        if self.id2word is None:
            raise ValueError("no word id mapping provided")
            # self.id2word = utils.dict_from_corpus(corpus)
            # self.num_terms = len(self.id2word)

        self.num_terms = 0 if not self.id2word else 1 + max(self.id2word.keys())

        if self.num_terms == 0:
            raise ValueError("empty collection (no terms)")

        self.num_topics: int = num_topics

        self.alpha: list[float] = [alpha] * num_topics
        self.beta: float = beta

        if prefix is None:
            rand_prefix = hex(random.randint(0, 0xFFFFFF))[2:] + '_'
            prefix = os.path.join(tempfile.gettempdir(), rand_prefix)

        self.prefix: str = prefix
        self.name: str = name
        self.twords: int = twords
        self.iterations: int = iterations
        self.sstep: int = sstep

        if corpus is not None:
            self.train(corpus)

    def topic_keys_filename(self):
        """Get path to topic keys text file."""
        return self.prefix + self.name + '.phi'

    def document_topics_filename(self):
        """Get path to document topic text file."""
        return self.prefix + self.name + '.theta'

    def text_corpus_filename(self):
        """Get path to corpus text file."""
        return self.prefix + self.name + '.corpus'

    def fvocabulary(self):
        """Get path to vocabulary text file."""
        return self.prefix + self.name + '.vocabulary'

    def ftopwords(self):
        """Get path to topwords text file."""
        return self.prefix + self.name + '.topWords'

    def word_weights_filename(self):
        """Get path to word weight file."""
        return self.prefix + 'wordweights.txt'

    def corpus2sttm(self, corpus: Iterable[tuple[int, int]], file_like):
        """Convert `corpus` to STTM format and write it to `file_like` descriptor.

        Format ::

            whitespace delimited utf8-encoded tokens[NEWLINE]

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Collection of texts in BoW format.
        file_like : file-like object
            Opened file.

        """
        for _, doc in enumerate(corpus):
            if self.id2word:
                tokens = chain.from_iterable([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc)
            else:
                tokens = chain.from_iterable([str(tokenid)] * int(cnt) for tokenid, cnt in doc)
            file_like.write(utils.to_utf8("{}\n".format(' '.join(tokens))))

    def convert_input(self, corpus: Iterable[tuple[int, int]]):
        """Convert corpus to SSTM format and save it to a temporary text file.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Collection of texts in BoW format.
        """
        logger.info("serializing temporary corpus to %s", self.text_corpus_filename())
        with smart_open(self.text_corpus_filename(), 'wb') as fout:
            self.corpus2sttm(corpus, fout)

    def train(self, corpus: Iterable[tuple[int, int]]):
        """Train STTM model.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format

        """
        self.convert_input(corpus)
        self.java_opts = '-Xmx1G'
        cmd = 'java {} -jar {} -model {} -corpus {} -ntopics {} -alpha {} -beta {} -niters {} -twords {} -name {} -sstep {}'
        cmd = cmd.format(
            self.java_opts,
            self.sstm_jar_path,
            self.model,
            self.text_corpus_filename(),
            self.num_topics,
            self.alpha[0],
            self.beta,
            self.iterations,
            self.twords,
            self.name,
            self.sstep,
        )

        if self.vectors is not None:
            cmd += ' -vectors {}'.format(self.vectors)

        logger.info("training STTM model with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        self.wordtopics = self.word_topics

    def __getitem__(self, bow, iterations=100):
        raise ValueError('infer-topics for new, unseen documents not implemented (or even possible?)')

    def load_word_topics(self):
        """Load words X topics matrix from :meth:`gensim.models.wrappers.ldamallet.LdaMallet.mallet_state_filename` file.

        Returns
        -------
        numpy.ndarray
            Matrix words X topics.

        """
        logger.info("loading assigned topics from %s", self.topic_keys_filename())

        # with open(self.topic_keys_filename(), 'r') as f:
        #    text = f.read().replace(' \n', '\n')
        # word_topics = np.loadtxt(io.StringIO(text), delimiter=' ', dtype=numpy.float64)

        word_topics = numpy.loadtxt(
            self.topic_keys_filename(), delimiter=' ', usecols=range(0, self.num_terms), dtype=numpy.float64
        )

        assert word_topics.shape == (self.num_topics, self.num_terms)

        return word_topics

    def load_document_topics(self):
        """Load document topics from :meth:`gensim.models.wrappers.ldamallet.LdaMallet.document_topics_filename` file.
        Shortcut for :meth:`gensim.models.wrappers.ldamallet.LdaMallet.read_doctopics`.

        Returns
        -------
        iterator of list of (int, float)
            Sequence of LDA vectors for documents.

        """
        return self.read_doctopics(self.document_topics_filename())

    def get_topics(self):
        """Get topics X words matrix.

        Returns
        -------
        numpy.ndarray
            Topics X words matrix, shape `num_topics` x `vocabulary_size`.

        """
        topics = self.word_topics
        return topics / topics.sum(axis=1)[:, None]

    def show_topics(self, num_topics: int = 10, num_words: int = 10):
        """Get the `num_words` most probable words for `num_topics` number of topics."""
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics: int = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics: int = min(num_topics, self.num_topics)
            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = self.alpha + 0.0001 * numpy.random.rand(len(self.alpha))
            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[: num_topics // 2] + sorted_topics[-num_topics // 2 :]
        shown = []
        for i in chosen_topics:
            topic: list[tuple] = self.show_topic(i, topn=num_words)
            shown.append((i, topic))
        return shown

    def show_topic(self, topicid, topn=10):
        """Get `num_words` most probable words for the given `topicid`."""

        if self.word_topics is None:
            logger.warning("Run train or load_word_topics before showing topics.")

        topic = self.word_topics[topicid]
        topic = topic / topic.sum()  # normalize to probability dist
        bestn = matutils.argsort(topic, topn, reverse=True)
        beststr = [(self.id2word[idx], topic[idx]) for idx in bestn]
        return beststr

    def read_doctopics(self, fname, eps=1e-6, renorm=True):
        """Get document topic vectors from MALLET's "doc-topics" format, as sparse gensim vectors.

        Parameters
        ----------
        fname : str
            Path to input file with document topics.
        eps : float, optional
            Threshold for probabilities.
        renorm : bool, optional
            If True - explicitly re-normalize distribution.

        Raises
        ------
        RuntimeError
            If any line in invalid format.

        Yields
        ------
        list of (int, float)
            LDA vectors for document.

        """
        doc_topics = numpy.loadtxt(fname, delimiter=' ', usecols=range(0, self.num_topics), dtype=numpy.float64)
        doc_topics[doc_topics < eps] = 0
        m = scipy.sparse.coo_matrix(doc_topics)

        for i in range(0, m.shape[0]):
            row = m.getrow(i)
            values = list(zip(row.indices, row.data))
            if renorm:
                total_weight = sum(w for _, w in values)
                if total_weight != 0:
                    values = [(i, w / total_weight) for (i, w) in values]
            yield values
