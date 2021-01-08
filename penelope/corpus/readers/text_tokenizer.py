import logging
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from penelope.utility import path_add_sequence, strip_path_and_extension
from penelope.vendor.nltk import word_tokenize

from .text_reader import TextReader, TextReaderOpts, TextSource
from .text_transformer import TextTransformOpts

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments,too-many-instance-attributes

FilenameOrCallableOrSequenceFilter = Union[Callable, Sequence[str]]

# TODO: Consider removing inheritence and instead use a TextReader as source?
# TODO: Move TextTransformOpts into this class?


class TextTokenizer(TextReader):
    """Reads a text corpus from `source` and applies given transforms.
    Derived classes can override `preprocess` as an initial step before transforms are applied.
    The `preprocess` is applied on the entire document, and the transforms on each token.
    The `preprocess can for instance be used to extract text from an XML file (see derived class SparvXmlCorpusSourceReader)
    """

    def __init__(
        self,
        source: TextSource,
        *,
        reader_opts: TextReaderOpts = None,
        transform_opts: TextTransformOpts = None,
        tokenize: Callable = None,
        chunk_size: int = None,
    ):
        """Derived class to TextReader that tokenizes and optionally chunks the text
            Also see :func:`~penelope.corpus.readers.TextReader`"

        Parameters
        ----------
        source : TextSource
            [description]
        reader_opts : str, optional
            [description], by default None
        transform_opts : TextTransformOpts
        tokenize : Callable, optional
            [description], by default None
        chunk_size : int, optional
            [description], by default None
        """
        reader_opts = reader_opts or TextReaderOpts()
        self._tokenize = tokenize or word_tokenize
        self.chunk_size = chunk_size
        super().__init__(source=source, reader_opts=reader_opts, transform_opts=transform_opts)

    def process(self, filename: str, content: str) -> Iterable[Tuple[str, List[str]]]:
        """Process a document and returns tokenized text, and optionally splits text in equal length chunks

        Parameters
        ----------
        content : str
            The actual text read from source.

        Yields
        -------
        Tuple[str,List[str]]
            Filename and tokens
        """
        for chunkname, text in super().process(filename, content):

            tokens = self._tokenize(text)

            filename = f"{strip_path_and_extension(chunkname)}.txt"

            if self.chunk_size is None:
                yield filename, tokens
            else:
                tokens = list(tokens)
                for n_chunk, i in enumerate(range(0, len(tokens), self.chunk_size)):
                    yield path_add_sequence(filename, n_chunk + 1, 3), tokens[i : i + self.chunk_size]
