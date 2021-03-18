# -*- coding: utf-8 -*-
import logging
import os

from lxml import etree

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))
XSLT_FILENAME = os.path.join(script_path, 'parlaclarin_xml_to_csv.xslt')

# pylint: disable=too-many-instance-attributes


def snuttify(token):
    if token is None:
        token = ''
    if token.startswith("'") and token.endswith("'"):
        return token
    return f"'{token}'"


class ParlaClarinXml2Csv:
    def __init__(
        self,
        xslt_filename: str = None,
        delimiter: str = "\t",
    ):
        self.xslt_filename = xslt_filename or XSLT_FILENAME
        self.xslt = etree.parse(self.xslt_filename)  # pylint: disable=I1101
        self.xslt_transformer = etree.XSLT(self.xslt)  # pylint: disable=I1101
        self.delimiter = snuttify(delimiter)

    def transform(self, content: str) -> str:
        xml = etree.XML(content)  # pylint: disable=I1101
        return self._transform(xml)

    def read_transform(self, filename: str) -> str:
        xml = etree.parse(filename)  # pylint: disable=I1101
        return self._transform(xml)

    def _transform(self, xml) -> str:
        text = self.xslt_transformer(xml, delimiter=self.delimiter)
        return str(text)
