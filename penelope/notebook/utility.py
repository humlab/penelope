import base64
from typing import Any

import ipywidgets as widgets
import pandas as pd
import yaml
from IPython.display import HTML
from IPython.display import display as ipython_display
from penelope.utility import getLogger

logger = getLogger()
# pylint: disable=too-many-ancestors


def create_download_link(df: pd.DataFrame, title: str = "Download CSV", filename: str = "data.csv") -> HTML:
    """Creates a download link for a Pandas dataframe without saving data to disk

        Source: https://medium.com/ibm-data-science-experience/how-to-upload-download-files-to-from-notebook-in-my-local-machine-6a4e65a15767

    Parameters
    ----------
    df : pd.DataFrame
    title : str, optional
        Link title, by default "Download CSV"
    filename : str, optional
        [description], by default "data.csv"

    Returns
    -------
    HTML
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload, title=title, filename=filename)
    return HTML(html)


class OutputsTabExt(widgets.Tab):
    def __init__(self, names):
        super().__init__()
        self.children = [widgets.Output() for _ in range(0, len(names))]
        self.loaded = [False for _ in range(0, len(names))]
        self.updaters = [None for _ in range(0, len(names))]
        _ = [self.set_title(i, name) for i, name in enumerate(names)]
        self.observe(self.on_tab_clicked, names='selected_index')

    def on_tab_clicked(self, widget):
        i = widget['new']
        if self.updaters[i] is not None:
            what, clear, plot = self.updaters[i]
            self.display_content(i, what=what, clear=clear, plot=plot)

    def display(self):
        ipython_display(self)
        return self

    def display_content(self, i: int, what: Any, clear: bool = False, plot=True):

        try:
            if clear:

                self.children[i].clear_output()
                self.loaded[i] = False

            with self.children[i]:

                if not self.loaded[i]:

                    if plot:
                        ipython_display(what if not callable(what) else what())
                    elif callable(what):
                        what()

                    self.loaded[i] = True

        except ValueError as ex:
            logger.error(f"display_content: index {i}, type{type(what)} failed: {str(ex)}. ")

        return self

    def display_fx_result(self, i, fx, *args, clear=False, lazy=False, plot=True, **kwargs):

        self.loaded[i] = False

        if lazy:
            self.updaters[i] = [lambda: fx(*args, **kwargs), clear, plot]
        else:
            self.display_content(i, lambda: fx(*args, **kwargs), clear=clear, plot=plot)

        return self

    def display_as_yaml(self, i: int, what: Any, clear: bool = False, width='800px', height='600px'):

        if what is None:
            logger.info(f"display_as_yaml: index {i} what is None")
            return self

        yaml_text = yaml.dump(what, explicit_start=True, line_break=True, indent=4)
        _what = widgets.Textarea(value=yaml_text, layout=widgets.Layout(width=width, height=height))
        self.display_content(i, _what, clear=clear, plot=True)
        return self
