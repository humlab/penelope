from dataclasses import dataclass, field
from typing import List

from penelope.co_occurrence import ContextOpts
from penelope.corpus import TokensTransformOpts, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts, TextReaderOpts
from penelope.pipeline.config import CorpusConfig, CorpusType
from penelope.pipeline.interfaces import PipelinePayload
from penelope.utility.pos_tags import PoS_Tag_Scheme, PoS_Tag_Schemes


def FakeSSI(source: str, index_source: str) -> CorpusConfig:
    corpus_config = CorpusConfig(
        corpus_name='test',
        corpus_type=CorpusType.Text,
        corpus_pattern="*.zip",
        language='english',
        text_reader_opts=TextReaderOpts(
            filename_fields=["unesco_id:_:2", "year:_:3", r'city:\w+\_\d+\_\d+\_\d+\_(.*)\.txt'],
            index_field=None,  # Use filename as key
            filename_filter=None,
            filename_pattern="*.txt",
            as_binary=False,
        ),
        pipeline_payload=PipelinePayload(
            source=None,
            document_index_source=None,
            document_index_key=None,
            document_index_sep=';',
            pos_schema_name="Universal",
            memory_store={
                'tagger': 'spaCy',
                'text_column': 'text',
                'pos_column': 'pos_',
                'lemma_column': 'lemma_',
                'spacy_model': "en_core_web_sm",
                'nlp': None,
                'lang': 'en',
            },
        ),
    )
    corpus_config.pipeline_payload.source = source
    corpus_config.pipeline_payload.document_index_source = index_source

    return corpus_config


@dataclass
class FakeGUI:  # pylint: disable=too-many-instance-attributes)

    corpus_tag: str
    corpus_filename: str
    target_folder: str = "./tests/output"

    pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
    tokens_transform_opts: TokensTransformOpts = TokensTransformOpts(
        extra_stopwords=[],
        keep_numerals=True,
        keep_symbols=True,
        language='english',
        max_len=None,
        min_len=1,
        only_alphabetic=False,
        only_any_alphanumeric=False,
        remove_accents=False,
        remove_stopwords=True,
        stopwords=None,
        to_lower=True,
        to_upper=False,
    )
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True,
        target_override=None,
        pos_includes='|NOUN|PROPN|VERB|',
        pos_excludes='|PUNCT|EOL|SPACE|',
        passthrough_tokens=[],
        append_pos=False,
    )
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = TaggedTokensFilterOpts(
        is_alpha=False,
        is_space=False,
        is_punct=False,
        is_digit=None,
        is_stop=None,
    )
    context_opts: ContextOpts = ContextOpts(
        context_width=4,
        concept={},
        ignore_concept=False,
    )
    count_threshold: int = 1
    partition_keys: List[str] = field(default_factory=lambda: ['year'])

    vectorize_opts: VectorizeOpts = VectorizeOpts(
        already_tokenized=True,
        lowercase=False,
        max_df=1.0,
        min_df=1,
        verbose=False,
    )
