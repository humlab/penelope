import csv
import io

import pandas as pd
import penelope.corpus.sparv.sparv_xml_to_csv as sparv

from ..interfaces import TextReaderOpts
from .reader import CorpusReader
from .sources import ZipSource


def create_sparv_xml_corpus_reader(
    source_path: str, reader_opts: TextReaderOpts, sparv_version: int = 4, content_type: str = 'pandas'
) -> CorpusReader:

    if not reader_opts.as_binary:
        raise ValueError("misconfiguration: XML files must be read in binary mode")

    if not reader_opts.filename_pattern.endswith('xml'):
        raise ValueError(f"misconfiguration: XML extension {reader_opts.filename_pattern} not expected")

    if sparv_version not in [3, 4]:
        raise ValueError(f"misconfiguration: sparv_version {sparv_version} not expected")

    if content_type not in ['xml', 'csv', 'pandas']:
        raise ValueError(f"misconfiguration: expected content_type xml, csv or pandas but found {content_type}")

    parser = sparv.SparvXml2CSV(delimiter="\t", version=sparv_version)

    def xml_to_csv_str(xml_doc: bytes) -> str:
        csv_doc = parser.transform(xml_doc)
        return csv_doc

    def xml_to_tagged_frame(xml_doc: bytes) -> pd.DataFrame:
        csv_doc = xml_to_csv_str(xml_doc)
        df = pd.read_csv(io.StringIO(csv_doc), sep='\t', quoting=csv.QUOTE_NONE)
        # FIXME: Read names from Config.payload.memory_store
        df.columns = ['text', 'lemma', 'pos']
        return df

    preprocess = xml_to_tagged_frame if content_type == 'pandas' else xml_to_csv_str if content_type == 'csv' else None

    source = ZipSource(source_path=source_path)
    reader = CorpusReader(source=source, reader_opts=reader_opts, preprocess=preprocess)

    return reader
