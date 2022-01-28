import ipywidgets as w

from penelope import topic_modelling as tm

from . import model_container as mc


class TopicsStateGui:
    def __init__(self, state: mc.TopicModelContainer) -> None:
        super().__init__()
        self.state: mc.TopicModelContainer = state

    @property
    def inferred_topics(self) -> tm.InferredTopicsData:
        return self.state["inferred_topics"]

    @property
    def inferred_n_topics(self) -> int:
        return self.inferred_topics.num_topics


class NextPrevTopicMixIn:
    def __init__(self, **kwargs) -> None:

        self._topic_id: w.IntSlider = w.IntSlider(min=0, max=199, step=1, value=0, continuous_update=False)

        button_style: dict = dict(description_width='initial', button_color='lightgreen')

        self._prev_topic_id: w.Button = w.Button(description="<<", layout=button_style)
        self._next_topic_id: w.Button = w.Button(description=">>", layout=button_style)

        super().__init__(**kwargs)

    def goto_previous(self, *_):
        self._topic_id.value = (self._topic_id.value - 1) % self._topic_id.max

    def goto_next(self, *_):
        self._topic_id.value = (self._topic_id.value + 1) % self._topic_id.max

    def setup(self, **kwargs) -> "NextPrevTopicMixIn":

        if hasattr(super(), "setup"):
            getattr(super(), "setup")(**kwargs)

        self._prev_topic_id.on_click(self.goto_previous)
        self._next_topic_id.on_click(self.goto_next)

        return self

    @property
    def topic_id(self) -> int:
        return self._topic_id.value


class AlertMixIn:
    def __init__(self, **kwargs) -> None:
        self._alert: w.HTML = w.HTML()
        super().__init__(**kwargs)

    def alert(self, msg: str):
        self._alert.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")
