from textacy import Corpus as TextacyCorpus


def infrequent_words(corpus: TextacyCorpus, normalize="lemma", weighting="count", threshold=0, as_strings=False):
    """Returns set of infrequent words i.e. words having total count less than given threshold"""

    if weighting == "count" and threshold <= 1:
        return set([])

    word_counts = corpus.word_counts(normalize=normalize, weighting=weighting, as_strings=as_strings)
    words = {w for w in word_counts if word_counts[w] < threshold}

    return words


def frequent_document_words(
    corpus: TextacyCorpus, normalize="lemma", weighting="freq", dfs_threshold=80, as_strings=True
):
    """Returns set of words that occurrs freuently in many documents, candidate stopwords"""
    document_freqs = corpus.word_doc_counts(
        normalize=normalize, weighting=weighting, smooth_idf=True, as_strings=as_strings
    )
    frequent_words = {w for w, f in document_freqs.items() if int(round(f, 2) * 100) >= dfs_threshold}
    return frequent_words
