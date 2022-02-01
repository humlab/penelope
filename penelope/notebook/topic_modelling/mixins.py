from __future__ import annotations

from typing import Any, Callable

import ipywidgets as w

from penelope import topic_modelling as tm

from .. import widgets_utils as wu
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

        # self._topic_id: w.IntSlider = w.IntSlider(min=0, max=199, step=1, value=0, continuous_update=False, description_width='initial')
        self._prev_topic_id: w.Button = w.Button(description="<<", layout=dict(button_style='Success', width="40px"))
        self._topic_id: w.Dropdown = w.Dropdown(options=[(str(i), i) for i in range(0, 200)], layout=dict(width="80px"))
        self._next_topic_id: w.Button = w.Button(description=">>", layout=dict(button_style='Success', width="40px"))
        self._next_prev_layout: w.HBox = w.HBox([self._prev_topic_id, self._topic_id, self._next_topic_id])
        self._prev_topic_id.style.button_color = 'lightgreen'
        self._next_topic_id.style.button_color = 'lightgreen'
        super().__init__(**kwargs)

    @property
    def __wc_max_topic_id(self) -> int:
        if isinstance(self._topic_id, w.Dropdown):
            return len(self._topic_id.options) - 1
        return self._topic_id.max

    def goto_previous(self, *_):
        self._topic_id.value = (self._topic_id.value - 1) % self.__wc_max_topic_id

    def goto_next(self, *_):
        self._topic_id.value = (self._topic_id.value + 1) % self.__wc_max_topic_id

    def setup(self, **kwargs) -> "NextPrevTopicMixIn":

        if hasattr(super(), "setup"):
            getattr(super(), "setup")(**kwargs)

        self._prev_topic_id.on_click(self.goto_previous)
        self._next_topic_id.on_click(self.goto_next)

        if hasattr(self, "inferred_topics"):
            self.topic_id = (0, getattr(self, "inferred_topics").n_topics - 1)

        return self

    @property
    def topic_id(self) -> tuple | int:
        return self._topic_id.value

    @topic_id.setter
    def topic_id(self, value: tuple | int) -> None:
        """Set current topic ID. If tuple (value, max) is given then both value and max are set"""
        if isinstance(value, tuple):
            if isinstance(self._topic_id, w.IntSlider):
                self._topic_id.value = value[0]
                self._topic_id.max = value[1]
            elif isinstance(self._topic_id, w.Dropdown):
                self._topic_id.value = None
                self._topic_id.options = [(str(i), i) for i in range(0, value[1])]
                self._topic_id.value = value[0]

        else:
            self._topic_id.value = value


class AlertMixIn:
    def __init__(self, **kwargs) -> None:
        self._alert: w.HTML = w.HTML()
        self._alert.value = "&nbsp;"

        super().__init__(**kwargs)

    def alert(self, msg: str):
        self._alert.value = msg or "&nbsp;"

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")

    def observe_slider_update_label(
        self, slider: w.IntSlider, label: w.HTML, text: str, decimals: int = 2
    ) -> Callable[[Any], None]:
        is_range: bool = isinstance(slider, (w.IntRangeSlider, w.FloatRangeSlider))
        is_float: bool = isinstance(slider, (w.FloatSlider, w.FloatRangeSlider))
        number_fmt: str = f"{{:.{decimals}f}}" if is_float else "{}"
        value_fmt: str = f"{number_fmt}-{number_fmt}" if is_range else number_fmt

        def get_label(value) -> str:
            tag: str = "" if slider.value is None else value_fmt.format(*value) if is_range else value_fmt.format(value)
            return f"<b>{text}</b> {tag}"

        def handler(*_):
            label.value = get_label(slider.value)

        label.value = get_label(slider.value)
        slider.observe(handler, names='value')
        # handler()


class ComputeMixIn:
    def __init__(self, **kwargs) -> None:
        self._compute: w.Button = w.Button(description='Show!', button_style='Success', layout={'width': '140px'})
        self._auto_compute: w.ToggleButton = w.ToggleButton(description="auto", value=False, layout={'width': '140px'})
        self._compute_handler: Callable[[Any], None] = None
        super().__init__(**kwargs)

    def update_handler(self, *_) -> None:  # pylint: disable=arguments-differ,unused-argument
        ...

    def observe(self, value: bool, handler: Any, **_) -> None:  # pylint: disable=arguments-differ,unused-argument
        ...

    def setup(self, **kwargs) -> ComputeMixIn:
        self._compute.on_click(self._mx_compute_handler_proxy)
        if hasattr(super(), "setup"):
            getattr(super(), "setup")(**kwargs)
        if not self._compute_handler:
            return self
        wu.register_observer(self._auto_compute, handler=self._auto_compute_handler, value=False)
        return self

    def _auto_compute_handler(self, *_):
        if self._compute_handler is not None:
            self._auto_compute.icon = 'check' if self.auto_compute else ''
            self._compute.disabled = self.auto_compute
            self.observe(value=self.auto_compute, handler=self._compute_handler)
            if self.auto_compute:
                self._mx_compute_handler_proxy()  # pylint: disable=not-callable
        else:
            self._compute.disabled = True
            self._auto_compute.disabled = True

    def _mx_compute_handler_proxy(self, *args) -> None:
        if self._compute_handler:
            self._compute_handler(args)  # pylint: disable=not-callable

    @property
    def auto_compute(self) -> bool:
        return self._compute_handler is not None and self._auto_compute.value
