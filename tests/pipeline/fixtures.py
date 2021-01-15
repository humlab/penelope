from penelope.co_occurrence import ContextOpts
from penelope.corpus import TokensTransformOpts, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts, TextReaderOpts
from penelope.notebook.interface import ComputeOpts
from penelope.pipeline.config import CorpusType


def FakeComputeOptsSpacyCSV(
    *,
    corpus_tag: str = 'MARS',
    corpus_filename: str = './tests/test_data/legal_instrument_five_docs_test.zip',
):  # pylint: disable=too-many-instance-attributes)

    return ComputeOpts(
        corpus_tag=corpus_tag,
        corpus_filename=corpus_filename,
        target_folder="./tests/output",
        corpus_type=CorpusType.SpacyCSV,
        # pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
        tokens_transform_opts=TokensTransformOpts(
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
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.csv',
            filename_fields=['year:_:1'],
            index_field=None,  # use filename
            as_binary=False,
        ),
        extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
            lemmatize=True,
            target_override=None,
            pos_includes='|NOUN|PROPN|VERB|',
            pos_excludes='|PUNCT|EOL|SPACE|',
            passthrough_tokens=[],
            append_pos=False,
        ),
        tagged_tokens_filter_opts=TaggedTokensFilterOpts(
            is_alpha=False,
            is_punct=False,
            is_digit=None,
            is_stop=None,
            is_space=False,
        ),
        create_subfolder=False,
        persist=True,
        context_opts=ContextOpts(
            context_width=4,
            concept={},
            ignore_concept=False,
        ),
        count_threshold=1,
        partition_keys=['year'],
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            max_df=1.0,
            min_df=1,
            verbose=False,
        ),
    )


def FakeComputeOptsSparvCSV(
    *,
    corpus_tag: str = 'TELLUS',
    corpus_filename: str = './tests/test_data/tranströmer_corpus_export.csv.zip',
) -> ComputeOpts:  # pylint: disable=too-many-instance-attributes)

    return ComputeOpts(
        corpus_tag=corpus_tag,
        corpus_filename=corpus_filename,
        target_folder="./tests/output",
        corpus_type=CorpusType.SparvCSV,
        tokens_transform_opts=TokensTransformOpts(
            to_lower=True,
            min_len=1,
            remove_stopwords=None,
            keep_symbols=True,
            keep_numerals=True,
            only_alphabetic=False,
            only_any_alphanumeric=False,
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.csv',
            filename_fields=('year:_:1',),
            index_field=None,  # use filename
            as_binary=False,
        ),
        extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
            pos_includes=None,
            pos_excludes='|MAD|MID|PAD|',
            lemmatize=False,
        ),
        tagged_tokens_filter_opts=TaggedTokensFilterOpts(
            is_alpha=False,
            is_punct=False,
            is_digit=None,
            is_stop=None,
            is_space=False,
        ),
        create_subfolder=False,
        persist=True,
        context_opts=ContextOpts(
            concept=('jag',),
            context_width=2,
        ),
        count_threshold=1,
        partition_keys=['year'],
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
        ),
    )
