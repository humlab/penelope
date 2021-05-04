from typing import Any
from unittest.mock import Mock, patch

import penelope.notebook.co_occurrence.to_co_occurrence_gui as to_co_occurrence_gui
import penelope.notebook.utility as notebook_utility
from penelope.co_occurrence import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.notebook import interface
from penelope.pipeline import CorpusConfig
from penelope.utility import PropertyValueMaskingOpts


def dummy_config():
    return CorpusConfig.load(path='./tests/test_data/SSI.yml')


@patch('penelope.notebook.utility.FileChooserExt2', Mock(spec=notebook_utility.FileChooserExt2))
def test_to_co_occurrence_create_gui():  # pylint: disable=unused-argument
    def done_callback(_: Any, __: interface.ComputeOpts):  # pylint: disable=unused-argument
        pass

    def compute_callback(args, corpus_config):  # pylint: disable=unused-argument
        pass

    corpus_config = dummy_config()

    gui = to_co_occurrence_gui.create_compute_gui(
        corpus_folder='./tests/test_data',
        data_folder='./tests/test_data',
        corpus_config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    assert gui is not None


@patch('penelope.notebook.utility.FileChooserExt2', Mock(spec=notebook_utility.FileChooserExt2))
def test_GUI_setup():  # pylint: disable=unused-argument
    def done_callback(*_, **__):
        pass

    def compute_callback(args, corpus_config):  # pylint: disable=unused-argument
        pass

    corpus_config = dummy_config()
    gui = to_co_occurrence_gui.ComputeGUI(
        default_corpus_path='./tests/test_data',
        default_corpus_filename='',
        default_data_folder='./tests/output',
    ).setup(
        config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    # layout = gui.layout()
    # gui._compute_handler({})  # pylint: disable=protected-access

    assert gui is not None


def test_generate_cli_opts():
    compute_opts = interface.ComputeOpts(
        corpus_type=interface.CorpusType.SparvCSV,
        corpus_filename="apa.txt",
        target_folder='/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/APA',
        corpus_tag='APA',
        tokens_transform_opts=TokensTransformOpts(
            only_alphabetic=False,
            only_any_alphanumeric=False,
            to_lower=True,
            to_upper=False,
            min_len=1,
            max_len=None,
            remove_accents=False,
            remove_stopwords=True,
            stopwords=None,
            extra_stopwords=['örn'],
            language='swedish',
            keep_numerals=True,
            keep_symbols=True,
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.csv',
            filename_filter=None,
            filename_fields=[
                'year:prot\\_(\\d{4}).*',
                'year2:prot_\\d{4}(\\d{2})__*',
                'number:prot_\\d+[afk_]{0,4}__(\\d+).*',
            ],
            index_field=None,
            as_binary=False,
            sep='\t',
            quoting=3,
        ),
        extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
            lemmatize=True,
            target_override=None,
            pos_includes="NN",
            pos_excludes=None,
            pos_paddings="MID|MAD|PAD",
            passthrough_tokens=[],
            append_pos=False,
        ),
        tagged_tokens_filter_opts=PropertyValueMaskingOpts(),
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            stop_words=None,
            max_df=1.0,
            min_df=1,
            verbose=False,
        ),
        count_threshold=1,
        create_subfolder=True,
        persist=True,
        context_opts=ContextOpts(
            context_width=1,
            concept=["apa"],
            ignore_concept=False,
        ),
        partition_keys=['year'],
        force=False,
    )

    cli_command: str = compute_opts.command_line("apa")

    assert cli_command is not None
