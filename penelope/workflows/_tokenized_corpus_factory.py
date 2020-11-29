from penelope.corpus import SparvTokenizedCsvCorpus, TextTransformOpts, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextTokenizer
from penelope.corpus.readers.interfaces import TextReaderOpts


def create_corpus(
    corpus_type: str,
    input_filename: str,
    tokens_transform_opts: TokensTransformOpts,
    reader_opts: TextReaderOpts,
    extract_tokens_opts: ExtractTaggedTokensOpts,
    chunk_size: int = None,
) -> TokenizedCorpus:
    return _ABSTRACT_FACTORY.get(corpus_type, NullCorpusFactory).create(
        input_filename=input_filename,
        tokens_transform_opts=tokens_transform_opts,
        reader_opts=reader_opts,
        extract_tokens_opts=extract_tokens_opts,
        chunk_size=chunk_size,
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
        reader_opts: TextReaderOpts,
        extract_tokens_opts: ExtractTaggedTokensOpts,  # pylint: disable=unused-argument
        chunk_size: int = None,
    ):
        corpus = TokenizedCorpus(
            TextTokenizer(
                source=input_filename,
                reader_opts=reader_opts,
                transform_opts=TextTransformOpts(
                    fix_whitespaces=True,
                    fix_hyphenation=True,
                ),
                chunk_size=chunk_size,
            ),
            tokens_transform_opts=tokens_transform_opts,
        )
        return corpus


class SparvTokenizedCsvCorpusFactory:
    @staticmethod
    def create(
        input_filename: str,
        tokens_transform_opts: TokensTransformOpts,
        reader_opts: TextReaderOpts,
        extract_tokens_opts: ExtractTaggedTokensOpts,
        chunk_size: int = None,
    ):
        corpus = SparvTokenizedCsvCorpus(
            source=input_filename,
            reader_opts=reader_opts,
            extract_tokens_opts=extract_tokens_opts,
            tokens_transform_opts=tokens_transform_opts,
            chunk_size=chunk_size,
        )
        return corpus


_ABSTRACT_FACTORY = {"text": TextTokenizedCorpusFactory, "sparv4-csv": SparvTokenizedCsvCorpusFactory}
