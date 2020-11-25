# -*- coding: utf-8 -*-
import logging

from penelope.corpus.sparv.sparv_xml_to_text import XSLT_FILENAME_V3, SparvXml2Text

from .interfaces import ExtractTokensOpts, TextSource
from .text_tokenizer import TextTokenizer
from .text_transformer import TextTransformer

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments, too-many-instance-attributes


class SparvXmlTokenizer(TextTokenizer):
    def __init__(
        self,
        source: TextSource,
        *,
        extract_tokens_opts: ExtractTokensOpts = None,
        xslt_filename: str = None,
        version: int = 4,
        **reader_opts,
    ):
        """Sparv XML file reader

        Parameters
        ----------
        source : TextSource
            Source (filename, ZIP, tokenizer)
        extract_tokens_opts : ExtractTokensOpts, optional
        xslt_filename : str, optional
            XSLT filename, by default None
        version : int, optional
            Sparv version, by default 4
        reader_opts : Dict[str, Any]
            chunk_size : int
                Optional chunking of text in chunk_size pieces
            filename_pattern : str
                Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Sequence[Sequence[IndexOfSplitOrCallableOrRegExp]]
                Document metadata fields to extract from filename
            filename_fields_key : str
                Field to be used as document_id
        """
        self.delimiter: str = ' '
        reader_opts = {
            **dict(
                filename_pattern='*.xml',
                tokenize=lambda x: x.split(self.delimiter),
                as_binary=True,
            ),
            **reader_opts,
        }
        super().__init__(source, **reader_opts)

        self.text_transformer = TextTransformer()
        self.extract_tokens_opts = extract_tokens_opts or ExtractTokensOpts()
        self.xslt_filename = XSLT_FILENAME_V3 if version == 3 else xslt_filename
        self.parser = SparvXml2Text(
            xslt_filename=self.xslt_filename,
            delimiter=self.delimiter,
            extract_tokens_opts=self.extract_tokens_opts,
        )

    def preprocess(self, content):
        return self.parser.transform(content)


class Sparv3XmlTokenizer(SparvXmlTokenizer):
    def __init__(
        self,
        source: TextSource,
        *,
        extract_tokens_opts: ExtractTokensOpts = None,
        **reader_opts,
    ):
        """Sparv v3 XML file reader

        Parameters
        ----------
        source : TextSource
            Source (filename, folder, zip, list)
        extract_tokens_opts: ExtractTokensOpts, optional
        reader_opts : Dict[str, Any]
            chunk_size : int
                Optional chunking of text in chunk_size pieces
            filename_pattern : str
                Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields_key : str
                Field to be used as document_id
         """
        super().__init__(
            source,
            extract_tokens_opts=extract_tokens_opts,
            xslt_filename=XSLT_FILENAME_V3,
            **reader_opts,
        )
