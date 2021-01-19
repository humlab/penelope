import csv
import logging
import os
from typing import Any, Dict, Set

from penelope.corpus.readers import ExtractTaggedTokensOpts

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))

# pylint: disable=too-many-instance-attributes


class SparvCsvToText:
    """Reads a Sparv CSV-file, applies filters and returns it as text"""

    def __init__(
        self,
        extract_tokens_opts: ExtractTaggedTokensOpts = None,
        delimiter: str = '\t',
        fields_index: Dict[str, int] = None,
    ):
        self.extract_tokens_opts = extract_tokens_opts
        self.delimiter = delimiter
        self.fields_index = fields_index or {'token': 0, 'pos': 1, 'baseform': 2}

    def transform(self, content: str):
        reader = csv.reader(content.splitlines(), delimiter=self.delimiter, quoting=csv.QUOTE_NONE)
        return self._transform(reader)

    def read_transform(self, filename: str) -> str:
        reader = csv.reader(filename, delimiter=self.delimiter, quoting=csv.QUOTE_NONE)
        return self._transform(reader)

    def _transform(self, reader: Any) -> str:  # Any = csv._reader
        _opts = self.extract_tokens_opts
        _lemmatize: bool = _opts.lemmatize
        _pos_includes: str = _opts.get_pos_includes()
        _pos_excludes: str = _opts.get_pos_excludes()
        _passthrough_tokens: Set[str] = _opts.get_passthrough_tokens()
        _append_pos: bool = _opts.append_pos

        _pos = self.fields_index['pos']
        _tok = self.fields_index['token']
        _lem = self.fields_index['baseform']

        data = (x for x in reader if len(x) == 3 and x[_tok] != '' and x[_pos] != '')

        next(data)

        if _pos_includes is not None:
            if len(_passthrough_tokens) == 0:
                data = (x for x in data if x[_pos] in _pos_includes)
            else:
                if _lemmatize:
                    data = (x for x in data if x[_lem] in _passthrough_tokens or x[_pos] in _pos_includes)
                else:
                    data = (x for x in data if x[_tok] in _passthrough_tokens or x[_pos] in _pos_includes)

        if _pos_excludes is not None:
            data = (x for x in data if x[_pos] not in _pos_excludes)

        if _lemmatize:
            if _append_pos:
                data = (
                    (f"{x[_tok] if x[_lem].strip('|') == '' else x[_lem].strip('|').split('|')[0]}|{x[_pos]}")
                    for x in data
                )
            else:
                data = ((x[_tok] if x[_lem].strip('|') == '' else x[_lem].strip('|').split('|')[0]) for x in data)
        else:
            if _append_pos:
                data = (f"{x[_tok]}|x{_pos}" for x in data)
            else:
                data = (x[_tok] for x in data)

        return ' '.join([x for x in data])
