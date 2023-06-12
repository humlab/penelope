from unittest.mock import patch

import pytest

try:
    from penelope.scripts import utils as script_utils

    CLICK_INSTALLED = True
except (ImportError, NameError):
    CLICK_INSTALLED = False


@pytest.mark.skipif(not CLICK_INSTALLED, reason="package `click` not installed")
@patch('penelope.scripts.utils.passed_cli_arguments', lambda _: dict())
def test_update_arguments_from_options_file_with_no_cli_override() -> dict:
    pytest.importorskip("click")
    yaml_data: str = "  - delta: 48"
    args: dict = dict(alfa=1, beta=2, delta=3, pi=3.14, options=yaml_data)
    args = script_utils.consolidate_arguments(arguments=args, filename_key='options')
    assert args['delta'] == 48


@pytest.mark.skipif(not CLICK_INSTALLED, reason="package `click` not installed")
@patch('penelope.scripts.utils.passed_cli_arguments', lambda _: dict(delta=999))
def test_update_arguments_from_options_file_with_cli_override() -> dict:
    pytest.importorskip("click")
    yaml_data: str = "  - delta: 48"
    args: dict = dict(alfa=1, beta=2, delta=3, pi=3.14, options=yaml_data)
    args = script_utils.consolidate_arguments(arguments=args, filename_key='options')
    assert args['delta'] == 999
