from unittest.mock import patch

import pytest

try:
    from penelope.scripts.co_occurrence import process_co_ocurrence
except (ImportError, NameError):
    ...


def monkey_patch(*_, **__):
    pass


@pytest.mark.skip("something is wrong")
@pytest.mark.long_running
@patch('penelope.co_occurrence.persistence.store_co_occurrences', monkey_patch)
def test_process_co_ocurrence():

    args: dict = dict(
        corpus_config="./tests/test_data/riksdagens-protokoll.yml",
        input_filename="./tests/test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip",
        output_filename="./tests/output/test_process_co_ocurrence",
        concept=None,
        ignore_concept=False,
        ignore_padding=False,
        context_width=6,
        compute_processes=None,
        compute_chunk_size=None,
        partition_key=["year"],
        phrase=None,
        phrase_file=None,
        create_subfolder=True,
        pos_includes="|NN|PM|VB|JJ|",
        pos_paddings="|UO|",
        pos_excludes="|MAD|MID|PAD|",
        to_lower=True,
        lemmatize=True,
        remove_stopwords='swedish',
        min_word_length=1,
        max_word_length=None,
        keep_symbols=True,
        keep_numerals=True,
        only_any_alphanumeric=False,
        only_alphabetic=False,
        tf_threshold=10,
        tf_threshold_mask=False,
        enable_checkpoint=True,
        force_checkpoint=False,
    )

    process_co_ocurrence(**args)


if __name__ == "__main__":
    test_process_co_ocurrence()
