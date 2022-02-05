from os.path import join as jj

import ipywidgets as w
import pandas as pd
from ipydatagrid import TextRenderer

from .. import grid_utility as nu
from . import mixins as mx
from . import model_container as mc


class EditTopicLabelsGUI(mx.AlertMixIn, mx.TopicsStateGui):
    def __init__(self, folder: str, state: mc.TopicModelContainer):
        super().__init__(state=state)
        self.folder = folder
        self._grid: nu.TableWidget = None
        self._save: w.Button = w.Button(description="Save")
        self._text: w.HTML = w.HTML()

    def setup(self) -> "EditTopicLabelsGUI":
        if 'label' not in self.inferred_topics.topic_token_overview.columns:
            default_labels: pd.Series = self.inferred_topics.topic_token_overview.index.astype(str)
            self.inferred_topics.topic_token_overview['label'] = default_labels
        self._grid = nu.table_widget(
            self.inferred_topics.topic_token_overview[['label']],
            handler=self.click_handler,
            editable=True,
        )
        self._grid.layout.height = "400px"
        self._grid.layout.width = "300px"
        self._grid.base_column_size = 207
        self._grid.auto_fit_params = {"area": "body"}
        self._grid.auto_fit_columns = False
        self._grid.renderers = {
            "label": TextRenderer(text_color="blue", background_color="white"),
        }
        self._text.layout.width = "600px"
        self._text.layout.height = "400px"
        self._grid.on_cell_change(self.on_cell_changed)
        self._save.on_click(self.save)
        return self

    def layout(self) -> w.Widget:
        return w.VBox(
            [
                w.HBox([self._save, self._alert]),
                w.HBox([self._grid, self._text]),
            ]
        )

    def on_cell_changed(self, _):
        self.alert("")
        self._save.disabled = False
        self._save.button_style = 'Success'

    def click_handler(self, row: pd.Series, grid):  # pylint: disable=unused-argument
        self.alert(f"<b>Topic ID</b>: {row.name}")
        self._text.value = self.inferred_topics.get_topic_title2(row.name, n_tokens=200)

    def save(self, *_):
        try:
            self.inferred_topics.topic_token_overview['label'] = self._grid.data['label']
            self.inferred_topics.topic_token_overview.to_csv(
                jj(self.folder, "topic_token_overview_label.csv"), sep='\t'
            )
            self._save.disabled = True
            self._save.button_style = ''
            self.alert(f'🙃 Saved to {jj(self.folder, "topic_token_overview_label.csv")}!')
        except Exception as ex:
            self.warn(f"😡 {ex}")
