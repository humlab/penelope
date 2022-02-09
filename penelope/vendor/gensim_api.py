
try:
    from gensim.corpora import MmCorpus
except ImportError:
    MmCorpus = None

try:
    from gensim.matutils import Sparse2Corpus
except ImportError:

    class Sparse2Corpus:
        def __init__(self, sparse, documents_columns=True):
            if documents_columns:
                self.sparse = sparse.tocsc()
            else:
                self.sparse = sparse.tocsr().T

        def __iter__(self):
            for indprev, indnow in zip(self.sparse.indptr, self.sparse.indptr[1:]):
                yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

        def __len__(self):
            return self.sparse.shape[1]

        def __getitem__(self, document_index):
            indprev = self.sparse.indptr[document_index]
            indnow = self.sparse.indptr[document_index + 1]
            return list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

