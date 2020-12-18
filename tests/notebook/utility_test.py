import pandas as pd
from IPython.display import Javascript
from penelope.notebook.utility import OutputsTabExt, create_js_download, shorten_path_with_ellipsis
from penelope.utility import getLogger

logger = getLogger()


def test_create_js_download():
    df = pd.DataFrame(data=dict(a=[1, 2], b=[1, 2]))
    w = create_js_download(df)
    assert isinstance(w, Javascript)


def test_output_tabs_ext():

    tab = OutputsTabExt(names=['A', 'B'])
    assert tab is not None
    assert len(tab.children) == 2
    assert len(tab.loaded) == 2


def test_shorten_path_with_ellipsis():
    path = '/abc/def/geh/ijk/mno'
    result = shorten_path_with_ellipsis(path, 10)
    assert len(result) <= 10 + len('.../')
