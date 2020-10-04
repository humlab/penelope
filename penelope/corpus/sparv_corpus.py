from __future__ import annotations

import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as tokenized_corpus
import penelope.utility.file_utility as file_utility
from penelope.corpus.tokens_transformer import default_opts


class SparvTokenizedCorpus(tokenized_corpus.TokenizedCorpus):

    def __init__(
        self,
        source,
        version,
        *,
        pos_includes=None,
        pos_excludes="|MAD|MID|PAD|",
        lemmatize=True,
        chunk_size=None,
        **tokens_transform_opts
    ):

        tokens_transform_opts = {k: v for k, v in tokens_transform_opts.items() if k in default_opts()}

        tokenizer = readers.SparvXmlTokenizer(
            source,
            transforms=None,
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
            lemmatize=lemmatize,
            chunk_size=chunk_size,
            xslt_filename=None,
            append_pos="",
            version=version
        )
        super().__init__(tokenizer, **tokens_transform_opts)


def sparv_extract_and_store(source: str, target: str, version: int, **opts):

    corpus = SparvTokenizedCorpus(source, version, **opts)

    file_utility.store(target, corpus)
