from unittest.mock import patch

from penelope.scripts import utils as script_utils


@patch('penelope.scripts.utils.passed_cli_arguments', lambda _: dict())
def test_update_arguments_from_options_file_with_no_cli_override() -> dict:
    yaml_data: str = "  - delta: 48"
    args: dict = dict(alfa=1, beta=2, delta=3, pi=3.14, options=yaml_data)
    args = script_utils.consolidate_cli_arguments(arguments=args, filename_key='options')
    assert args['delta'] == 48


@patch('penelope.scripts.utils.passed_cli_arguments', lambda _: dict(delta=999))
def test_update_arguments_from_options_file_with_cli_override() -> dict:
    yaml_data: str = "  - delta: 48"
    args: dict = dict(alfa=1, beta=2, delta=3, pi=3.14, options=yaml_data)
    args = script_utils.consolidate_cli_arguments(arguments=args, filename_key='options')
    assert args['delta'] == 999
