import csv
import logging
import os
from typing import Any, Dict, Set

from penelope.corpus.readers import ExtractTaggedTokensOpts

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))

# pylint: disable=too-many-instance-attributes


# TODO: Consolidate with tagged_frame_to_tokens? Same business logic.
class SparvCsvToText:
    """Reads a Sparv CSV-file, applies filters and returns it as text"""

    def __init__(
        self,
        extract_tokens_opts: ExtractTaggedTokensOpts = None,
        delimiter: str = '\t',
        fields_index: Dict[str, int] = None,
    ):
        self.extract_tokens_opts: ExtractTaggedTokensOpts = extract_tokens_opts
        self.delimiter: str = delimiter
        self.fields_index: Dict[str, int] = fields_index or {
            extract_tokens_opts.text_column: 0,
            extract_tokens_opts.pos_column: 1,
            extract_tokens_opts.lemma_column: 2,
        }

    def transform(self, content: str):
        reader = csv.reader(content.splitlines(), delimiter=self.delimiter, quoting=csv.QUOTE_NONE)
        return self._transform(reader)

    def read_transform(self, filename: str) -> str:
        reader = csv.reader(filename, delimiter=self.delimiter, quoting=csv.QUOTE_NONE)
        return self._transform(reader)

    def _transform(self, reader: Any) -> str:  # Any = csv._reader
        _opts = self.extract_tokens_opts
        _lemmatize: bool = _opts.lemmatize
        _pos_includes: Set[str] = _opts.get_pos_includes()
        _pos_paddings: Set[str] = _opts.get_pos_paddings()
        _pos_excludes: Set[str] = _opts.get_pos_excludes()
        _passthrough_tokens: Set[str] = _opts.get_passthrough_tokens()
        _block_tokens: Set[str] = _opts.get_block_tokens()
        _append_pos: bool = _opts.append_pos
        _pad: str = "*"

        _pos = self.fields_index[self.extract_tokens_opts.pos_column]
        _tok = self.fields_index[self.extract_tokens_opts.text_column]
        _lem = self.fields_index[self.extract_tokens_opts.lemma_column]

        data = (x for x in reader if len(x) == 3 and x[_tok] != '' and x[_pos] != '')

        next(data)

        _pos_all_includes: Set[str] = _pos_includes.union(_pos_paddings)

        if len(_pos_includes) > 0:
            """Don't filter if PoS-include is empty - and don't filter out PoS tokens that should be padded"""
            if len(_passthrough_tokens) == 0:
                data = (x for x in data if x[_pos] in _pos_all_includes)
            data = (x for x in data if x[_lem] in _passthrough_tokens or x[_pos] in _pos_all_includes)

        if _block_tokens:
            data = (x for x in data if x[_lem] not in _block_tokens)

        if _pos_excludes is not None:
            data = (x for x in data if (x[_pos] not in _pos_excludes or x[_lem] in _passthrough_tokens))

        if _lemmatize:
            if _append_pos:
                data = (
                    _pad
                    if x[_pos] in _pos_paddings
                    else f"{x[_tok] if x[_lem].strip('|') == '' else x[_lem].strip('|').split('|')[0]}@{x[_pos]}"
                    for x in data
                )
            else:
                if len(_pos_paddings) > 0:
                    data = (
                        (
                            _pad
                            if x[_pos] in _pos_paddings
                            else x[_tok]
                            if x[_lem].strip('|') == ''
                            else x[_lem].strip('|').split('|')[0]
                        )
                        for x in data
                    )
                else:
                    data = ((x[_tok] if x[_lem].strip('|') == '' else x[_lem].strip('|').split('|')[0]) for x in data)
        else:
            data = (
                (_pad if x[_pos] in _pos_paddings else f"{x[_tok]}@x{_pos}" for x in data)
                if _append_pos
                else (_pad if x[_pos] in _pos_paddings else x[_tok] for x in data)
            )

        return ' '.join([x.replace(" ", "_") for x in data])
