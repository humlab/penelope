import pytest
from penelope.scripts.co_occurrence import process_co_ocurrence


@pytest.mark.skip(reason="Long running")
def test_process_co_ocurrence():

    args: dict = dict(
        corpus_config="./tests/test_data/riksdagens-protokoll.yml",
        input_filename="./tests/test_data/riksdagens-protokoll.1920-2019.test.zip",
        output_filename="./tests/output/test_process_co_ocurrence",
        concept=["information"],
        no_concept=False,
        context_width=6,
        partition_key=["year"],
        create_subfolder=True,
        pos_includes="|NN|PM|UO|",
        pos_excludes="|MAD|MID|PAD|",
        to_lowercase=True,
        lemmatize=True,
        remove_stopwords='swedish',
        min_word_length=1,
        max_word_length=None,
        keep_symbols=True,
        keep_numerals=True,
        only_any_alphanumeric=False,
        only_alphabetic=False,
        count_threshold=10,
    )

    process_co_ocurrence(**args)


if __name__ == "__main__":
    test_process_co_ocurrence()
