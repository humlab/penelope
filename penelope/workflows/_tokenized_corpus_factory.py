from typing import Dict

from penelope.corpus import SparvTokenizedCsvCorpus, TextTransformOpts, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import ExtractTokensOpts, TextTokenizer


def create_corpus(
    corpus_type: str,
    input_filename: str,
    tokens_transform_opts: TokensTransformOpts,
    reader_opts: Dict,
    extract_tokens_opts: ExtractTokensOpts,
    **_,
) -> TokenizedCorpus:
    return _ABSTRACT_FACTORY.get(corpus_type, NullCorpusFactory).create(
        input_filename=input_filename,
        tokens_transform_opts=tokens_transform_opts,
        reader_opts=reader_opts,
        extract_tokens_opts=extract_tokens_opts,
    )


class NullCorpusFactory:
    @staticmethod
    def create(*_):
        raise ValueError("Abstract factory: unknown corpus type")


class TextTokenizedCorpusFactory:
    @staticmethod
    def create(
        input_filename: str,
        tokens_transform_opts: TokensTransformOpts,
        reader_opts: Dict,
        extract_tokens_opts: ExtractTokensOpts,  # pylint: disable=unused-argument
    ):
        corpus = TokenizedCorpus(
            TextTokenizer(
                source=input_filename,
                **reader_opts,
                text_transform_opts=TextTransformOpts(
                    fix_whitespaces=True,
                    fix_hyphenation=True,
                ),
            ),
            tokens_transform_opts=tokens_transform_opts,
        )
        return corpus


class SparvTokenizedCsvCorpusFactory:
    @staticmethod
    def create(
        input_filename: str,
        tokens_transform_opts: TokensTransformOpts,
        reader_opts: Dict,
        extract_tokens_opts: ExtractTokensOpts,  # pylint: disable=unused-argument
    ):
        corpus = SparvTokenizedCsvCorpus(
            source=input_filename,
            reader_opts=reader_opts,
            extract_tokens_opts=extract_tokens_opts,
            tokens_transform_opts=tokens_transform_opts,
        )
        return corpus


_ABSTRACT_FACTORY = {"text": TextTokenizedCorpusFactory, "sparv4-csv": SparvTokenizedCsvCorpusFactory}
