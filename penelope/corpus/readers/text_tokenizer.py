import logging
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from penelope.corpus.readers.interfaces import TextSource
from penelope.corpus.readers.text_reader import TextReader
from penelope.utility import IndexOfSplitOrCallableOrRegExp, strip_path_and_extension
from penelope.utility.filename_utils import path_add_sequence
from penelope.vendor.nltk import word_tokenize

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
    The `preprocess can for instance be used to extract text from an XML file (see derived class SParvXmlCorpusSourceReader)
    """

    def __init__(
        self,
        source: TextSource,
        *,
        filename_pattern: str = None,
        filename_filter: FilenameOrCallableOrSequenceFilter = None,
        filename_fields: Sequence[IndexOfSplitOrCallableOrRegExp] = None,
        as_binary: bool = False,
        transforms: List[Callable] = None,
        text_transform_opts: TextTransformOpts = None,
        tokenize: Callable = None,
        chunk_size: int = None,
    ):
        """Derived class to TextReader that tokenizes and optionally chunks the text
            Also see :func:`~penelope.corpus.readers.TextReader`"

        Parameters
        ----------
        source : TextSource
            [description]
        transforms : List[Callable], optional
            [description], by default None
        chunk_size : int, optional
            [description], by default None
        filename_pattern : str, optional
            [description], by default None
        filename_filter : Union[Callable, List[str]], optional
            [description], by default None
        filename_fields : Sequence[IndexOfSplitOrCallableOrRegExp], optional
            [description], by default None
        tokenize : Callable, optional
            [description], by default None
        as_binary : bool, optional
            [description], by default False
        text_transform_opts : TextTransformOpts
        """
        self._tokenize = tokenize or word_tokenize
        self.chunk_size = chunk_size
        super().__init__(
            source=source,
            filename_pattern=filename_pattern,
            filename_filter=filename_filter,
            filename_fields=filename_fields,
            extra_text_transforms=transforms,
            text_transform_opts=text_transform_opts,
            as_binary=as_binary,
        )

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
        for ubername, text in super().process(filename, content):

            tokens = self._tokenize(text)

            basename = f"{strip_path_and_extension(ubername)}.txt"

            if self.chunk_size is None:
                yield basename, tokens
            else:
                tokens = list(tokens)
                for n_chunk, i in enumerate(range(0, len(tokens), self.chunk_size)):
                    yield path_add_sequence(basename, n_chunk + 1, 3), tokens[i : i + self.chunk_size]
