# type: ignore

try:
    from spacy import load
    from spacy.language import Language
    from spacy.tokens import Doc, Token

    __spacy_installed: bool = True
except (ImportError, NameError):
    __spacy_installed: bool = False

try:
    from .utility import SPACY_DATA, load_model, load_model_by_parts, prepend_path, prepend_spacy_path, token_count_by
except (ImportError, NameError):
    ...
