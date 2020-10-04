import pytest  # pylint: disable=unused-import

from penelope.corpus.text_transformer import TextTransformer


@pytest.mark.xfail
def transform_smoke_test():
    transformer = TextTransformer()

    assert transformer is not None
