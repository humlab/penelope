import stanza

def download_model(lang: str):
    try:
        nlp = stanza.Pipeline(lang)
        del nlp
    except Exception:
        stanza.download(lang)