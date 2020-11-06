import pytest  # pylint: disable=unused-import
from penelope.corpus.tokens_transformer import TokensTransformer, TokensTransformOpts


@pytest.mark.xfail
def transform_smoke_test():
    transformer = TokensTransformer(tokens_transform_opts=TokensTransformOpts())

    assert transformer is not None
