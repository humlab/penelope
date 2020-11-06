# -*- coding: utf-8 -*-
import logging

from penelope.corpus.sparv.sparv_csv_to_text import SparvCsvToText

from .interfaces import FilenameOrFolderOrZipOrList
from .option_objects import AnnotationOpts
from .text_tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments


class SparvCsvTokenizer(TextTokenizer):
    def __init__(
        self,
        source: FilenameOrFolderOrZipOrList,
        *,
        annotation_opts: AnnotationOpts = None,
        **tokenizer_opts,
    ):
        """[summary]

        Parameters
        ----------
        source : [type]
            [description]
        annotation_opts : AnnotationOpts, optional
        tokenizer_opts : Dict[str, Any]
            Optional chunking of text in chunk_size pieces
            filename_pattern : str
            Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Sequence[Sequence[IndexOfSplitOrCallableOrRegExp]]
                Document metadata fields to extract from filename
            as_binary : bool
                Open input file as binary file (XML)

        """
        self.delimiter: str = '\t'

        super().__init__(
            source,
            **{**dict(tokenize=lambda x: x.split(), filename_pattern='*.csv', transforms=None), **tokenizer_opts},
        )

        self.annotation_opts = annotation_opts or AnnotationOpts()
        self.parser = SparvCsvToText(delimiter=self.delimiter, annotation_opts=self.annotation_opts)

    def preprocess(self, content):
        return self.parser.transform(content)
