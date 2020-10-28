import pytest
import textacy
from textacy import Corpus

import penelope.vendor.textacy as textacy_utility


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module")
def mary_had_a_little_lamb_corpus():
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
    corpus = Corpus("en", data=texts)
    return corpus


@pytest.fixture(scope="module")
def mary_had_a_little_lamb_tokens():
    tokenized_docs = [
        list(doc._.to_terms_list(ngrams=1, entities=None, normalize="lower", as_strings=True))
        for doc in mary_had_a_little_lamb_corpus()
    ]
    return tokenized_docs


def test_pos_extract_when_pos_includes_is_noun(mary_had_a_little_lamb_corpus: textacy.Corpus):

    terms = (
        textacy_utility.ExtractPipeline(mary_had_a_little_lamb_corpus, target='lemma')
        .pos(
            include_pos=(
                'NOUN',
                'PROPN',
            )
        )
        .process()
    )

    terms = [d for d in terms]

    assert ["Mary", "lamb", "fleece", "snow"] == terms[0]


def test_pos_extract_when_pos_includes_is_jj(mary_had_a_little_lamb_corpus: textacy.Corpus):

    terms = (
        textacy_utility.ExtractPipeline(mary_had_a_little_lamb_corpus, target='lemma')
        .pos(include_pos=('ADJ',))
        .process()
    )

    terms = [d for d in terms]
    assert ["little", "white"] == terms[0]


def test_pos_extract_when_a_more_complexed_filter(mary_had_a_little_lamb_corpus: textacy.Corpus):

    terms = (
        textacy_utility.ExtractPipeline(mary_had_a_little_lamb_corpus, target='lemma')
        .pos(include_pos=('NOUN', 'PROPN'))
        .remove_stopwords(extra_stopwords=[])
        .attributes_filter(filter_nums=True, filter_punct=True)
        .infrequent_word_filter(1)
        .frequent_word_filter(100)
        .min_character_filter(2)
        .substitute(subst_map={'Mary': 'mar4'})
        .predicate(predicate=lambda x: True)
        .transform(transformer=lambda x: x.upper())
        .process()
    )

    terms = [d for d in terms]
    assert ["MAR4", "LAMB", "FLEECE", "SNOW"] == terms[0]
