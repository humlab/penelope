from typing import Dict

from penelope.corpus import TextTransformOpts, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import TextTokenizer
from penelope.corpus.readers.option_objects import AnnotationOpts
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus


def create_corpus(
    corpus_type: str,
    input_filename: str,
    tokens_transform_opts: TokensTransformOpts,
    tokenizer_opts: Dict,
    annotation_opts: AnnotationOpts,
    **_,
) -> TokenizedCorpus:
    return _ABSTRACT_FACTORY.get(corpus_type, NullCorpusFactory).create(
        input_filename=input_filename,
        tokens_transform_opts=tokens_transform_opts,
        tokenizer_opts=tokenizer_opts,
        annotation_opts=annotation_opts,
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
        tokenizer_opts: Dict,
        annotation_opts: AnnotationOpts,  # pylint: disable=unused-argument
    ):
        corpus = TokenizedCorpus(
            TextTokenizer(
                source=input_filename,
                **tokenizer_opts,
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
        tokenizer_opts: Dict,
        annotation_opts: AnnotationOpts,  # pylint: disable=unused-argument
    ):
        corpus = SparvTokenizedCsvCorpus(
            source=input_filename,
            tokenizer_opts=tokenizer_opts,
            annotation_opts=annotation_opts,
            tokens_transform_opts=tokens_transform_opts,
        )
        return corpus


_ABSTRACT_FACTORY = {"text": TextTokenizedCorpusFactory, "sparv4-csv": SparvTokenizedCsvCorpusFactory}
