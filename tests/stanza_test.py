import stanza
from penelope.vendor.stanza.utility import download_model

# stanza.download('en')


def test_can_create_stanza_en_pipeline():

    download_model('en')
    _ = stanza.Pipeline('en')

    # en_nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse', verbose=False, use_gpu=False)
