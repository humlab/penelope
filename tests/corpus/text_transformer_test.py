import pytest  # pylint: disable=unused-import

from penelope.corpus.readers import TextTransformer
from penelope.corpus.transforms import normalize_characters


@pytest.mark.xfail
def transform_smoke_test():
    transformer = TextTransformer()

    assert transformer is not None


def test_normalize_characters():

    text = "räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗RÄKSMÖRGÅS"
    normalized_text = normalize_characters(text)
    assert normalized_text == 'räksmörgås----------++//~~~~~~~\'\'\'\'\'\'\'\'\'""""`′′′′′′RÄKSMÖRGÅS'

    text = "räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗RÄKSMÖRGÅS"
    normalized_text = normalize_characters(text, groups="double_quotes,tildes")
    assert normalized_text == 'räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕~~~~~~~’՚Ꞌꞌ＇‘’‚‛""""´″‴‵‶‷⁗RÄKSMÖRGÅS'
