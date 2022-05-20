import pandas as pd
import pytest

from penelope.vendor.gensim_api._gensim.wrappers.mallet_tm import MalletTopicModel


@pytest.mark.skip(reason="Bug fixed")
def test_diagnostics_to_topic_token_weights_data_bug_check():

    filename: str = '/data/westac/blm/mallet/100/mallet/diagnostics.xml'

    ttd: pd.DataFrame = MalletTopicModel.load_topic_token_diagnostics2(filename)

    assert ttd is not None

    nan_token_ids: int = len(ttd[ttd.token == 'None'])

    assert nan_token_ids == 0

    # ttd.to_csv("diagnostics_original_modified.csv", sep='\t')
