import csv
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))

# pylint: disable=too-many-instance-attributes


class SparvCsvToText:
    """Reads a Sparv CSV-file, applies filters and returns it as text"""

    def __init__(
        self,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: bool = True,
        delimiter: str = '\t',
        append_pos: bool = False,
        **fields,
    ):

        self.delimiter = delimiter
        self.lemmatize = lemmatize
        self.pos_includes = pos_includes.strip('|').split('|') if pos_includes is not None else None
        self.pos_excludes = pos_excludes.strip('|').split('|') if pos_excludes is not None else None
        self.append_pos = append_pos
        self.fields = fields or {'token': 0, 'pos': 1, 'baseform': 2}

    def transform(self, content: str):
        reader = csv.reader(content.splitlines(), delimiter=self.delimiter, quoting=csv.QUOTE_NONE)
        return self._transform(reader)

    def read_transform(self, filename: str):
        reader = csv.reader(filename, delimiter=self.delimiter, quoting=csv.QUOTE_NONE)
        return self._transform(reader)

    def _transform(self, reader: Any):  # Any = csv._reader

        _pos = self.fields['pos']
        _tok = self.fields['token']
        _lem = self.fields['baseform']

        data = (x for x in reader if len(x) == 3 and x[_tok] != '' and x[_pos] != '')

        next(data)

        if self.pos_includes is not None:
            data = (x for x in data if x[_pos] in self.pos_includes)

        if self.pos_excludes is not None:
            data = (x for x in data if x[_pos] not in self.pos_excludes)

        if self.lemmatize:
            if self.append_pos:
                data = (
                    (f"{x[_tok] if x[_lem].strip('|') == '' else x[_lem].strip('|').split('|')[0]}|{x[_pos]}")
                    for x in data
                )
            else:
                data = ((x[_tok] if x[_lem].strip('|') == '' else x[_lem].strip('|').split('|')[0]) for x in data)
        else:
            if self.append_pos:
                data = (f"{x[_tok]}|x{_pos}" for x in data)
            else:
                data = (x[_tok] for x in data)

        return ' '.join([x for x in data])
