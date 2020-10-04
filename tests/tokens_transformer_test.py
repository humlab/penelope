import pytest  # pylint: disable=unused-import

from penelope.corpus.tokens_transformer import TokensTransformer


@pytest.mark.xfail
def transform_smoke_test():
    transformer = TokensTransformer()

    assert transformer is not None
