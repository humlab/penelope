import io

import pandas as pd
from penelope.corpus.readers import TaggedTokensFilterOpts


def test_mask_punct_space_when_no_space():

    # data = doc.head().to_csv(sep='\t',header=True))
    data = """
\ttext\tlemma_\tpos_\tis_space\tis_punct
0\tConstitution\tconstitution\tNOUN\tFalse\tFalse
1\tof\tof\tADP\tFalse\tFalse
2\tthe\tthe\tDET\tFalse\tFalse
3\tUnited\tUnited\tPROPN\tFalse\tFalse
4\tNations\tNations\tPROPN\tFalse\tFalse
"""
    doc: pd.DataFrame = pd.read_csv(io.StringIO(data), sep='\t', index_col=0)

    filter_opts = TaggedTokensFilterOpts(is_punct=False, is_space=False)
    mask = filter_opts.mask(doc)

    assert mask is not None


def test_mask_punct_space_when_no_space_or_punct():
    data = {
        'index': [0, 1, 2, 3, 4, 5, 6, 7],
        'columns': ['text', 'lemma_', 'pos_', 'is_space', 'is_punct'],
        'data': [
            ['Mamma', 'mamma', 'NN', False, False],
            ['pappa', 'pappa', 'NN', False, False],
            ['varför', 'varför', 'HA', False, False],
            ['är', 'vara', 'VB', False, False],
            ['det', 'den', 'PN', False, False],
            ['så', 'så', 'AB', False, False],
            ['kallt', 'kall', 'JJ', False, False],
            ['?', '?', 'MAD', False, True],
        ],
    }

    doc = pd.DataFrame(**data)

    assert TaggedTokensFilterOpts(is_punct=False, is_space=False).mask(doc).sum() == 7


def test_mask_when_empty_document_succeeds():
    data = {
        'index': [],
        'columns': ['text', 'lemma_', 'pos_', 'is_space', 'is_punct'],
        'data': [],
    }

    doc = pd.DataFrame(**data)

    assert TaggedTokensFilterOpts(is_punct=False, is_space=False).mask(doc).sum() == 0
