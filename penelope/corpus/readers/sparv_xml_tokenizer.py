# -*- coding: utf-8 -*-
import logging

from penelope.corpus.sparv.sparv_xml_to_text import XSLT_FILENAME_V3, SparvXml2Text

from .interfaces import FilenameOrFolderOrZipOrList
from .option_objects import AnnotationOpts
from .text_tokenizer import TextTokenizer
from .text_transformer import TextTransformer

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments, too-many-instance-attributes


class SparvXmlTokenizer(TextTokenizer):
    def __init__(
        self,
        source: FilenameOrFolderOrZipOrList,
        *,
        annotation_opts: AnnotationOpts = None,
        xslt_filename: str = None,
        version: int = 4,
        **tokenizer_opts,
    ):
        """Sparv XML file reader

        Parameters
        ----------
        source : FilenameOrFolderOrZipOrList
            Source (filename, ZIP, tokenizer)
        annotation_opts : AnnotationOpts, optional
        xslt_filename : str, optional
            XSLT filename, by default None
        version : int, optional
            Sparv version, by default 4
        tokenizer_opts : Dict[str, Any]
            chunk_size : int
                Optional chunking of text in chunk_size pieces
            filename_pattern : str
                Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Sequence[Sequence[IndexOfSplitOrCallableOrRegExp]]
                Document metadata fields to extract from filename
        """
        self.delimiter: str = ' '
        tokenizer_opts = {
            **dict(
                filename_pattern='*.xml',
                tokenize=lambda x: x.split(self.delimiter),
                as_binary=True,
            ),
            **tokenizer_opts,
        }
        super().__init__(source, **tokenizer_opts)

        self.text_transformer = TextTransformer(transforms=[])
        self.annotation_opts = annotation_opts or AnnotationOpts()
        self.xslt_filename = XSLT_FILENAME_V3 if version == 3 else xslt_filename
        self.parser = SparvXml2Text(
            xslt_filename=self.xslt_filename,
            delimiter=self.delimiter,
            annotation_opts=self.annotation_opts,
        )

    def preprocess(self, content):
        return self.parser.transform(content)


class Sparv3XmlTokenizer(SparvXmlTokenizer):
    def __init__(
        self,
        source: FilenameOrFolderOrZipOrList,
        *,
        annotation_opts: AnnotationOpts = None,
        **tokenizer_opts,
    ):
        """Sparv v3 XML file reader

        Parameters
        ----------
        source : FilenameOrFolderOrZipOrList
            Source (filename, folder, zip, list)
        annotation_opts: AnnotationOpts, optional
        tokenizer_opts : Dict[str, Any]
            chunk_size : int
                Optional chunking of text in chunk_size pieces
            filename_pattern : str
                Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Sequence[Sequence[IndexOfSplitOrCallableOrRegExp]]
                Document metadata fields to extract from filename
        """
        super().__init__(
            source,
            annotation_opts=annotation_opts,
            xslt_filename=XSLT_FILENAME_V3,
            **tokenizer_opts,
        )
