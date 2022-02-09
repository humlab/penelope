from ..gensim_api import MmCorpus

class ExtMmCorpus(MmCorpus):
    """Extension of MmCorpus that allow TF normalization based on document length."""

    @staticmethod
    def norm_tf_by_D(doc):
        D = sum([x[1] for x in doc])
        return doc if D == 0 else map(lambda tf: (tf[0], tf[1] / D), doc)

    def __init__(self, fname):
        MmCorpus.__init__(self, fname)

    def __iter__(self):
        for doc in MmCorpus.__iter__(self):
            yield self.norm_tf_by_D(doc)

    def __getitem__(self, docno):
        return self.norm_tf_by_D(MmCorpus.__getitem__(self, docno))
