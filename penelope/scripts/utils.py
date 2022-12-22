import os
import sys
from os.path import dirname, isdir, isfile
from typing import Any, Callable, Optional

from penelope import pipeline, utility

try:
    import click
except ImportError:
    click = object()

# pylint: disable=no-member

CLI_LOG_PATH = './logs'

CLI_OPTIONS = {
    '--alpha': dict(help='Prior belief of topic probability. symmetric/asymmetric/auto', default='asymmetric'),
    '--append-pos': dict(help='Append PoS to tokens', default=False, is_flag=True),
    '--compute-chunk-size': dict(help='Compute process chunk size', default=10, type=click.INT),
    '--compute-processes': dict(help='Number of compute processes', default=None, type=click.INT),
    '--concept': dict(help='Concept', default=None, multiple=True, type=click.STRING),
    '--context-width': dict(
        help='Width of context on either side of concept. Window size = 2 * context_width + 1 ',
        default=None,
        type=click.INT,
    ),
    '--corpus-config': dict(help='Corpus config filename/folder', default=None),
    '--input-filename': dict(help='Corpus filename/folder (overrides config if specified)', default=None),
    '--output-filename': dict(help='Output filename (overrides config if specified)', default=None),
    '--corpus-folder': dict(help='Corpus folder (if vectorized corpus)'),
    '--create-subfolder': dict(
        help='Create subfolder in target folder named `target_name`', default=True, is_flag=True
    ),
    '--deserialize-processes': dict(
        help='Number of processes during deserialization', default=4, type=click.IntRange(1, 99)
    ),
    '--doc-chunk-size': dict(help='Split document in chunks of chunk-size words.', default=None, type=click.INT),
    '--enable-checkpoint/--no-enable-checkpoint': dict(help='Enable checkpoints', default=True, is_flag=True),
    '--engine': dict(help='LDA implementation', type=click.STRING),
    '--filename-pattern': dict(help='Filename pattern', default=None, type=click.STRING),
    '--filename-field': dict(help='Field(s) to extract from document name', type=click.STRING),
    '--fix-accents/--no-fix-accents': dict(help='Fix accents', default=False, is_flag=True),
    '--fix-hyphenation/--no-fix-hyphenation': dict(help='Fix hyphens', default=True, is_flag=True),
    '--force-checkpoint/--no-force-checkpoint': dict(
        help='Force new checkpoints (if enabled)', default=False, is_flag=True
    ),
    '--ignore-concept': dict(help='Filter out word pairs that include a concept token', default=False, is_flag=True),
    '--ignore-padding': dict(help='Filter out word pairs that include a padding token', default=False, is_flag=True),
    '--keep-numerals/--no-keep-numerals': dict(help='Keep numerals', default=True, is_flag=True),
    '--keep-symbols/--no-keep-symbols': dict(help='Keep symbols', default=True, is_flag=True),
    '--lemmatize/--no-lemmatize': dict(help='Use word baseforms', is_flag=True, default=True),
    '--max-iter': dict(help='Max number of iterations.', default=None, type=click.INT),
    '--num-top-words': dict(help='Number of word per topic to collect after estimation.', default=500, type=click.INT),
    '--max-tokens': dict(help='Only consider the `max-tokens` most frequent tokens', default=None, type=click.INT),
    '--max-word-length': dict(help='Max length of words to keep', default=None, type=click.IntRange(10, 99)),
    '--min-word-length': dict(help='Min length of words to keep', default=1, type=click.IntRange(1, 99)),
    '--minimum-probability': dict(
        help='Document-topic weights lower than value are discarded.', default=0.005, type=click.FloatRange(0.001, 0.10)
    ),
    '--n-topics': dict(help='Number of topics.', default=50, type=click.INT),
    '--n-tokens': dict(help='Number of tokens per topic.', default=None, type=click.INT),
    '--only-alphabetic': dict(help='Keep only tokens having only alphabetic characters', default=False, is_flag=True),
    '--only-any-alphanumeric': dict(
        help='Keep tokens with at least one alphanumeric char', default=False, is_flag=True
    ),
    '--options-filename': dict(
        help='Use values in YAML file as command line options.', type=click.STRING, default=None
    ),
    '--partition-key': dict(help='Partition key(s)', default=None, multiple=True, type=click.STRING),
    '--passes': dict(help='Number of passes.', default=None, type=click.INT),
    '--passthrough-column': dict(help="Use tagged columns as-is (ignore filters)", default=None, type=click.STRING),
    '--phrase-file': dict(help='Phrase filename', default=None, type=click.STRING),
    '--phrase': dict(help='Phrase', default=None, multiple=True, type=click.STRING),
    '--pos-excludes': dict(help='POS tags to exclude.', default='', type=click.STRING),
    '--pos-includes': dict(help='POS tags to include e.g. "|NN|JJ|".', default='', type=click.STRING),
    '--pos-paddings': dict(help='POS tags to replace with a padding marker.', default='', type=click.STRING),
    '--random-seed': dict(help="Random seed value", default=None, type=click.INT),
    '--remove-stopwords': dict(
        help='Remove stopwords using given language', default=None, type=click.Choice(['swedish', 'english'])
    ),
    '--store-compressed/--no-store-compressed': dict(
        help='Store training corpus compressed', default=True, is_flag=True
    ),
    '--store-corpus/--no-store-corpus': dict(help='Store training corpus', default=True, is_flag=True),
    '--target-folder': dict(help='Target folder, if none then `corpus-folder/target-name`.', type=click.STRING),
    '--target-mode': dict(
        help='What to do: train, predict or both', type=click.Choice(['train', 'predict', 'both']), default=None
    ),
    '--trained-model-folder': dict(
        help='If `target-mode` is `predict`, then folder with existing TM model', type=click.STRING, default=None
    ),
    '--tf-threshold-mask': dict(
        help='If true, then low TF words are kept, but masked as "__low_tf__"',
        default=False,
        is_flag=True,
        type=click.IntRange(1, 99),
    ),
    '--tf-threshold': dict(
        help='Globoal TF threshold filter (words below filtered out)', default=1, type=click.IntRange(1, 99)
    ),
    '--windows-threshold': dict(
        help='Globoal common windows count threshold (word-pairs below are filtered out)',
        default=1,
        type=click.IntRange(1, 99),
    ),
    '--to-lower/--no-to-lower': dict(help='Lowercase words', default=True, is_flag=True),
    '--train-corpus-folder': dict(help='Use train corpus in folder if exists', default=None, type=click.STRING),
    '--update-every': dict(
        help='Gensim specific, update every n interval of chunks dispatch.', default=1, type=click.INT
    ),
    '--chunk-size': dict(
        help='Gensim specific. Number of docs in each chunk (if online).', default=2000, type=click.INT
    ),
    '--workers': dict(help='Number of workers (if applicable).', default=None, type=click.INT),
    '--per-word-topics': dict(help='Compute word topic probabilities', default=False, is_flag=True),
}


def consolidate_cli_arguments(*, arguments: dict, filename_key: str, log_args: bool = True) -> dict:
    """Updates `arguments` based on values found in file specified by `filename_key`.
    Values specified at the command line overrides values from options file."""
    options_filename: Optional[str] = arguments.get(filename_key)
    del arguments[filename_key]
    arguments = utility.update_dict_from_yaml(options_filename, arguments)
    cli_args: dict = passed_cli_arguments(arguments)
    arguments.update(cli_args)

    if log_args:
        log_arguments(arguments)

    return arguments


def log_arguments(args: dict, subdir: bool = False) -> None:

    cli_command: str = utility.strip_path_and_extension(sys.argv[0])

    log_dir: str = os.path.join(CLI_LOG_PATH, cli_command) if subdir else CLI_LOG_PATH

    os.makedirs(log_dir, exist_ok=True)

    log_name: str = utility.ts_data_path(log_dir, f"{cli_command}.yml")

    utility.write_yaml(args, log_name)


def passed_cli_arguments(args: dict) -> dict:
    """Return args specified in commande line"""
    cli_args = {
        name: args[name]
        for name in args
        if click.get_current_context().get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
    }

    return cli_args


def load_config(config_filename: str, corpus_source: str) -> pipeline.CorpusConfig:
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(path=config_filename)
    if config.pipeline_payload.source is None:
        config.pipeline_payload.source = corpus_source
        if isdir(corpus_source):
            config.folders(corpus_source, method='replace')
        elif isfile(corpus_source):
            config.folders(dirname(corpus_source), method='replace')
    return config


def remove_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def option2(*param_decls: str, **attrs: Any) -> Callable[..., Any]:
    for opt_attrib in ('default', 'help', 'type', 'is_flag', 'multiple'):
        if opt_attrib not in attrs and any(p in CLI_OPTIONS for p in param_decls):
            opt_name: str = next(p for p in param_decls if p in CLI_OPTIONS)
            opt: dict = CLI_OPTIONS[opt_name]
            if opt_attrib in opt:
                attrs[opt_attrib] = opt[opt_attrib]
    return click.option(*param_decls, **attrs)
