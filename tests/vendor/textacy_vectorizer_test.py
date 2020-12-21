import numpy as np
import penelope.vendor.textacy as textacy_utility
import pytest
import textacy
from penelope.corpus import CorpusVectorizer, VectorizedCorpus


@pytest.fixture(scope="module")
def mary_had_a_little_lamb_corpus() -> textacy.Corpus:
    """Source: https://github.com/chartbeat-labs/textacy/blob/master/tests/test_vsm.py """
    texts = [
        "Mary had a little lamb. Its fleece was white as snow.",
        "Everywhere that Mary went the lamb was sure to go.",
        "It followed her to school one day, which was against the rule.",
        "It made the children laugh and play to see a lamb at school.",
        "And so the teacher turned it out, but still it lingered near.",
        "It waited patiently about until Mary did appear.",
        "Why does the lamb love Mary so? The eager children cry.",
        "Mary loves the lamb, you know, the teacher did reply.",
    ]
    corpus = textacy.Corpus("en", data=texts)
    return corpus


def test_vectorizer(mary_had_a_little_lamb_corpus: textacy.Corpus):  # pylint: disable=redefined-outer-name

    expected_dtm = np.matrix(
        [
            [0, 0, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 1],
        ]
    )

    terms = (
        textacy_utility.ExtractPipeline(mary_had_a_little_lamb_corpus, target='lemma')
        .pos(include_pos=('NOUN', 'PROPN'))
        .remove_stopwords(extra_stopwords=[])
        .ingest(filter_nums=True, filter_punct=True)
        .min_character_filter(2)
        .transform(transformer=lambda x: x.lower())
        .process()
    )

    document_terms = ((f'document_{i}.txt', tokens) for i, tokens in enumerate(terms))
    vectorizer = CorpusVectorizer()

    v_corpus: VectorizedCorpus = vectorizer.fit_transform(document_terms, already_tokenized=True)

    assert v_corpus is not None

    assert {
        'mary': 4,
        'lamb': 3,
        'fleece': 2,
        'snow': 7,
        'school': 6,
        'day': 1,
        'rule': 5,
        'child': 0,
        'teacher': 8,
    } == v_corpus.token2id

    assert (expected_dtm == v_corpus.data.todense()).all()
