# type: ignore
# pylint: disable=unused-argument, unused-import

from typing import Any

try:
    SPACY_INSTALLED: bool = True

    from spacy.language import Language
    from spacy.tokens import Doc, Token

    from ._spacy import (
        check_model_version,
        load,
        load_model,
        load_model_by_parts,
        prepend_path,
        prepend_spacy_path,
        spacy_data_path,
        spacy_version,
        token_count_by,
    )

except (ImportError, NameError):
    SPACY_INSTALLED: bool = False

    def stub(*_, **__):
        raise ModuleNotFoundError("Spacy is not installed")

    load = stub
    load_model = stub
    load_model_by_parts = stub
    prepend_path = stub
    prepend_spacy_path = stub
    token_count_by = stub
    Language = Any
    check_model_version = stub
    spacy_version = '0.0.0'
