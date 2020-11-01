import re

import penelope.utility as utility
import spacy
import textacy
from spacy.language import Language

logger = utility.getLogger('corpus_text_analysis')

LANGUAGE_MODEL_MAP = {'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm', 'it': 'it_core_web_sm', 'de': 'de_core_web_sm'}

_load_spacy = (
    textacy.load_spacy_lang if hasattr(textacy, 'load_spacy_lang') else textacy.load_spacy  # pylint: disable=no-member
)


def keep_hyphen_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return spacy.tokenizer.Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=None,
    )


@utility.timecall
def create_nlp(language, **nlp_args) -> Language:

    if len(nlp_args.get('disable', [])) == 0:
        nlp_args.pop('disable')

    def remove_whitespace_entities(doc):
        doc.ents = [e for e in doc.ents if not e.text.isspace()]
        return doc

    logger.info('Loading model: %s...', language)

    Language.factories['remove_whitespace_entities'] = lambda _nlp, **_cfg: remove_whitespace_entities
    model_name = LANGUAGE_MODEL_MAP[language]
    # if not model_name.endswith('lg'):
    #    logger.warning('Selected model is not the largest availiable.')

    nlp = _load_spacy(model_name, **nlp_args)
    nlp.tokenizer = keep_hyphen_tokenizer(nlp)

    pipeline = lambda: [x[0] for x in nlp.pipeline]

    logger.info('Using pipeline: %s', ' '.join(pipeline()))

    return nlp
