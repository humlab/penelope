# type: ignore
# pylint: disable=unused-argument, unused-import

try:
    import spacy

    SPACY_INSTALLED: bool = True
except (ImportError, NameError):
    SPACY_INSTALLED: bool = False
try:
    from ._spacy import (
        SPACY_DATA,
        Doc,
        Language,
        Token,
        load,
        load_model,
        load_model_by_parts,
        prepend_path,
        prepend_spacy_path,
        token_count_by,
    )
except (ImportError, NameError):
    ...
