# type: ignore
# pylint: disable=unused-import

from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import gensim.models as models
    from gensim.models import CoherenceModel
    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldamulticore import LdaMulticore
    from gensim.models.lsimodel import LsiModel

    from ._gensim.wrappers import LdaMallet

    __has_gensim: bool = True
except (ImportError, NameError):
    __has_gensim: bool = False
    logger.info("gensim not included in current installment")

# from ._gensim.ext_mm_corpus import ExtMmCorpus
try:
    from ._gensim.ext_text_corpus import ExtTextCorpus, SimpleExtTextCorpus
except (ImportError, NameError):
    ...

try:
    from ._gensim.utils import (
        from_id2token_to_dictionary,
        from_stream_of_tokens_to_dictionary,
        from_stream_of_tokens_to_sparse2corpus,
        from_token2id_to_dictionary,
    )
except (ImportError, NameError):
    ...

try:
    from ._gensim.wrappers import MalletTopicModel, STTMTopicModel
except (ImportError, NameError):
    ...

try:

    from gensim.corpora import MmCorpus
    from gensim.corpora.dictionary import Dictionary
    from gensim.corpora.textcorpus import TextCorpus

except (ImportError, NameError):

    MmCorpus = object
    TextCorpus = object

    class Dictionary(dict):
        @staticmethod
        def from_corpus(corpus, id2word=None):
            raise ModuleNotFoundError()


try:
    from gensim.matutils import Sparse2Corpus, corpus2csc
except (ImportError, NameError):

    class Sparse2Corpus:
        def __init__(self, sparse, documents_columns=True):
            self.sparse = sparse.tocsc() if documents_columns else sparse.tocsr().T

        def __iter__(self):
            for indprev, indnow in zip(self.sparse.indptr, self.sparse.indptr[1:]):
                yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

        def __len__(self):
            return self.sparse.shape[1]

        def __getitem__(self, document_index):
            indprev = self.sparse.indptr[document_index]
            indnow = self.sparse.indptr[document_index + 1]
            return list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def corpus2csc(corpus, num_terms=None, dtype=np.float64, num_docs=None, num_nnz=None, printprogress=0):
        raise ModuleNotFoundError("gensim not included in package")


try:
    from gensim.utils import check_output
except (ImportError, NameError):
    from subprocess import check_output
