# -*- coding: utf-8 -*-
import logging

from penelope.corpus.sparv.sparv_xml_to_text import XSLT_FILENAME_V3, SparvXml2Text
from penelope.corpus.text_transformer import TextTransformer

from .interfaces import FilenameOrFolderOrZipOrList
from .text_tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments, too-many-instance-attributes


class SparvXmlTokenizer(TextTokenizer):
    def __init__(
        self,
        source: FilenameOrFolderOrZipOrList,
        *,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: bool = True,
        xslt_filename: str = None,
        append_pos: bool = "",
        version: int = 4,
        **tokenizer_opts,
    ):
        """Sparv XML file reader

        Parameters
        ----------
        source : FilenameOrFolderOrZipOrList
            Source (filename, ZIP, tokenizer)
        pos_includes : str, optional
            POS to includde e.g. `|VB|NN|`, by default None
        pos_excludes : str, optional
            POS to exclude, by default "|MAD|MID|PAD|"
        lemmatize : bool, optional
            If True then return word baseform, by default True
        xslt_filename : str, optional
            XSLT filename, by default None
        append_pos : bool, optional
           If True them append POS to word, by default ""
        version : int, optional
            Sparv version, by default 4
        tokenizer_opts : Dict[str, Any]
            chunk_size : int
                Optional chunking of text in chunk_size pieces
            filename_pattern : str
                Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Dict[str,Union[Callable,str]]
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
        self.lemmatize = lemmatize
        self.append_pos = append_pos
        self.pos_includes = pos_includes
        self.pos_excludes = pos_excludes
        self.xslt_filename = XSLT_FILENAME_V3 if version == 3 else xslt_filename
        self.parser = SparvXml2Text(
            xslt_filename=self.xslt_filename,
            delimiter=self.delimiter,
            pos_includes=self.pos_includes,
            lemmatize=self.lemmatize,
            append_pos=self.append_pos,
            pos_excludes=self.pos_excludes,
        )

    def preprocess(self, content):
        return self.parser.transform(content)


class Sparv3XmlTokenizer(SparvXmlTokenizer):
    def __init__(
        self,
        source: FilenameOrFolderOrZipOrList,
        *,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: str = True,
        append_pos: str = "",
        **tokenizer_opts,
    ):
        """Sparv v3 XML file reader

        Parameters
        ----------
        source : FilenameOrFolderOrZipOrList
            Source (filename, folder, zip, list)
        pos_includes : str, optional
            POS to includde e.g. `|VB|NN|`, by default None
        pos_excludes : str, optional
            POS to exclude, by default "|MAD|MID|PAD|"
        lemmatize : bool, optional
            If True then return word baseform, by default True
        append_pos : bool, optional
           If True them append POS to word, by default ""
        tokenizer_opts : Dict[str, Any]
            chunk_size : int
                Optional chunking of text in chunk_size pieces
            filename_pattern : str
                Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Dict[str,Union[Callable,str]]
                Document metadata fields to extract from filename
        """
        super().__init__(
            source,
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
            lemmatize=lemmatize,
            xslt_filename=XSLT_FILENAME_V3,
            append_pos=append_pos,
            **tokenizer_opts,
        )
