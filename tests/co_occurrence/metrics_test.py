

from penelope.corpus.token2id import Token2Id
import pytest
import scipy
from penelope.co_occurrence.persistence import Bundle, to_filename
import numpy as np
import pandas as pd
from penelope.co_occurrence.hal_or_glove.vectorizer_hal import HyperspaceAnalogueToLanguageVectorizer
from penelope.corpus import VectorizedCorpus

def test_burgess_litmus_test():
    terms = 'The Horse Raced Past The Barn Fell .'.lower().split()
    answer = {
        'barn': {'.': 4, 'barn': 0, 'fell': 5, 'horse': 0, 'past': 0, 'raced': 0, 'the': 0},
        'fell': {'.': 5, 'barn': 0, 'fell': 0, 'horse': 0, 'past': 0, 'raced': 0, 'the': 0},
        'horse': {'.': 0, 'barn': 2, 'fell': 1, 'horse': 0, 'past': 4, 'raced': 5, 'the': 3},
        'past': {'.': 2, 'barn': 4, 'fell': 3, 'horse': 0, 'past': 0, 'raced': 0, 'the': 5},
        'raced': {'.': 1, 'barn': 3, 'fell': 2, 'horse': 0, 'past': 5, 'raced': 0, 'the': 4},
        'the': {'.': 3, 'barn': 6, 'fell': 4, 'horse': 5, 'past': 3, 'raced': 4, 'the': 2},
    }
    df_answer = pd.DataFrame(answer).astype(np.uint32)[['the', 'horse', 'raced', 'past', 'barn', 'fell']].sort_index()
    # display(df_answer)
    vectorizer = HyperspaceAnalogueToLanguageVectorizer()
    vectorizer.fit([terms], size=5, distance_metric=0)
    df_imp = vectorizer.to_df().astype(np.uint32)[['the', 'horse', 'raced', 'past', 'barn', 'fell']].sort_index()
    assert df_imp.equals(df_answer), "Test failed"
    # df_imp == df_answer

    # Example in Chen, Lu:
    terms = 'The basic concept of the word association'.lower().split()
    vectorizer = HyperspaceAnalogueToLanguageVectorizer().fit([terms], size=5, distance_metric=0)
    df_imp = vectorizer.to_df().astype(np.uint32)[['the', 'basic', 'concept', 'of', 'word', 'association']].sort_index()
    df_answer = pd.DataFrame(
        {
            'the': [2, 5, 4, 3, 6, 4],
            'basic': [3, 0, 5, 4, 2, 1],
            'concept': [4, 0, 0, 5, 3, 2],
            'of': [5, 0, 0, 0, 4, 3],
            'word': [0, 0, 0, 0, 0, 5],
            'association': [0, 0, 0, 0, 0, 0],
        },
        index=['the', 'basic', 'concept', 'of', 'word', 'association'],
        dtype=np.uint32,
    ).sort_index()[['the', 'basic', 'concept', 'of', 'word', 'association']]
    assert df_imp.equals(df_answer), "Test failed"
    print('Test run OK')




@pytest.fixture(scope="module")
def bundle() -> Bundle:
    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = to_filename(folder=folder, tag=tag)

    bundle: Bundle = Bundle.load(filename, compute_frame=False)

    return bundle


def compute_hal_score(corpus: VectorizedCorpus, bundle: Bundle) -> VectorizedCorpus:
    """Compute yearly HAL-score for each co-occurrence pair (w1, w2)

    HAL-score = (CW(w1)

    """


    return corpus

def test_compute_hal_score(bundle: Bundle):

    ...

    # nw_xy is co_occurrence_matrx

    # Must calculate nw_xy nw_x and nw_y for each year

    # nw_xy is given by co_occurrence matrix/corpus
    # nw_x is given by co_occurrence token windows count matrix

    token2id: Token2Id = bundle.token2id
    document_index: pd.DataFrame = bundle.document_index

    nw_xy: scipy.sparse.spmatrix = bundle.corpus.data
    nw_x: scipy.sparse.spmatrix = bundle.window_counts.document_counts

    assert nw_xy.shape == (len(document_index), len(bundle.corpus.token2id))

    assert nw_xy is not None
    assert nw_x is not None

