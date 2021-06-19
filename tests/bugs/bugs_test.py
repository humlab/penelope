import pytest
from penelope.scripts.co_occurrence import process_co_ocurrence

'''
co_occurrence --context-width 2 --ignore-padding --tf-threshold 3 --tf-threshold-mask --pos-includes "|NN|PM|PC|VB|" --pos-paddings
"|JJ|AB|HA|IE|IN|PL|KN|SN|RG|RO|UO|PP|DT|HD|HP|HS|PN|PS|" --pos-excludes "|MAD|MID|PAD|" --lemmatize --to-lowercase --keep-symbols
--keep-numerals --enable-checkpoint --force-checkpoint doit.yml
/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip
/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/APA/APA_co-occurrence.csv.zip
'''

@pytest.mark.long_running
def test_CLI_process_co_ocurrence():

    process_co_ocurrence(
        corpus_config="./doit.yml",
        input_filename='./tests/test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip',
        output_filename='./tests/output/APA/APA_co-occurrence.csv.zip',
        filename_pattern=None,
        concept=None,
        ignore_concept=False,
        ignore_padding=True,
        context_width=2,
        phrase=None,
        phrase_file=None,
        partition_key=None,
        create_subfolder=True,
        pos_includes="|NN|PM|PC|VB|",
        pos_paddings="PASSTHROUGH",
        pos_excludes="|MAD|MID|PAD|",
        append_pos=False,
        to_lowercase=True,
        lemmatize=True,
        remove_stopwords=False,
        min_word_length=1,
        max_word_length=None,
        keep_symbols=True,
        keep_numerals=True,
        only_any_alphanumeric=False,
        only_alphabetic=False,
        tf_threshold=10,
        tf_threshold_mask=False,
        enable_checkpoint=True,
        force_checkpoint=True,
    )

    assert True
