# import os

# import stanza


# def download_model(lang: str):
#     try:
#         nlp = stanza.Pipeline(lang)
#         del nlp
#     except Exception:
#         stanza.download(lang)


# def howto_install_corenlp(folder: str = "/data/vendor/corenlp"):
#     stanza.install_corenlp(dir=folder)


# def howto_download_corenlp_models(model: str, version: str, folder: str = None):
#     """Example how models can be fownloaded: """
#     if not folder:
#         if "CORENLP_HOME" in os.environ:
#             folder = os.environ["CORENLP_HOME"]
#         else:
#             FileNotFoundError("Either supply target folder or set CORENLP_HOME environ variable")

#     stanza.download_corenlp_models(model=model, version=version, dir=folder)
