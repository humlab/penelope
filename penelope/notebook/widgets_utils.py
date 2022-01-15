import contextlib
from typing import Any, Callable, List, Sequence

import ipywidgets as widgets
import numpy as np
from bokeh.models import ColumnDataSource, CustomJS

AGGREGATES = {'mean': np.mean, 'sum': np.sum, 'max': np.max, 'std': np.std}

default_graph_tools = "pan,wheel_zoom,box_zoom,reset,hover,save"

MATPLOTLIB_PLOT_STYLES = [
    'ggplot',
    'bmh',
    'seaborn-notebook',
    'seaborn-whitegrid',
    '_classic_test',
    'seaborn',
    'fivethirtyeight',
    'seaborn-white',
    'seaborn-dark',
    'seaborn-talk',
    'seaborn-colorblind',
    'seaborn-ticks',
    'seaborn-poster',
    'seaborn-pastel',
    'fast',
    'seaborn-darkgrid',
    'seaborn-bright',
    'Solarize_Light2',
    'seaborn-dark-palette',
    'grayscale',
    'seaborn-muted',
    'dark_background',
    'seaborn-deep',
    'seaborn-paper',
    'classic',
]


def display_text_on_hover_js_code(
    element_id: str, id_name: str, text_name: str, glyph_name: str = 'glyph', glyph_data: str = 'glyph_data'
) -> str:
    return f"""
        const indices = cb_data.index.indices;
        var current_id = -1;
        if (indices.length > 0) {{
            var index = indices[0];
            var id = parseInt({glyph_name}.data.{id_name}[index]);
            if (id !== current_id) {{
                current_id = id;
                const text = {glyph_data}.data.{text_name}[id];
                document.getElementsByClassName('{element_id}')[0].innerText = 'ID ' + id.toString() + ': ' + text;
            }}
        }}
    """


def glyph_hover_callback2(
    glyph_source: ColumnDataSource, glyph_id: str, text_ids: List[str], text: Sequence[str], element_id: str
):
    """Glyph hover callback that displays the text associated with glyph's id in text element `text_id`"""
    text_source: ColumnDataSource = ColumnDataSource(dict(text_id=text_ids, text=text))
    return glyph_hover_callback(glyph_source, glyph_id, text_source, element_id)


def glyph_hover_callback(
    glyph_source: ColumnDataSource, glyph_id: str, text_source: ColumnDataSource, element_id: str
) -> CustomJS:
    code: str = display_text_on_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
    callback: CustomJS = CustomJS(args={'glyph': glyph_source, 'glyph_data': text_source}, code=code)
    return callback


BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')


def button_with_callback(description: str, style: widgets.Layout = None, callback: Callable = None):
    style = style or BUTTON_STYLE
    btn = widgets.Button(description=description, style=style)
    if callback is not None:
        btn.on_click(callback)
    return btn


def text_widget(element_id: str = None, default_value: str = '') -> widgets.HTML:
    """Adds an HTML span with element id `element_id` (can be referenced from JS)."""
    value = "<span class='{}'>{}</span>".format(element_id, default_value) if element_id is not None else ''
    return widgets.HTML(value=value, placeholder='', description='')


def button_with_next_callback(that_with_property: Any, property_name: str, count: int) -> None:
    def f(_):
        control = getattr(that_with_property, property_name, None)
        if control is not None:
            control.value = (control.value + 1) % count

    return button_with_callback(description=">>", callback=f)


def button_with_previous_callback(that_with_property: Any, property_name: str, count: int) -> None:
    def f(_):
        control = getattr(that_with_property, property_name, None)
        if control is not None:
            control.value = (control.value - 1) % count

    return button_with_callback(description="<<", callback=f)


def register_observer(ctrl: widgets.Widget, *, handler: Callable, value: bool, names: str = 'value') -> None:
    method: str = 'observe' if value else 'unobserve'
    if hasattr(ctrl, method):
        with contextlib.suppress(ValueError):
            getattr(ctrl, method)(handler, names=names)


# import asyncio
# from time import time

# def throttle(wait):
#     """ Decorator that prevents a function from being called
#         more than once every wait period. """
#     def decorator(fn):
#         time_of_last_call = 0
#         scheduled, timer = False, None
#         new_args, new_kwargs = None, None
#         def throttled(*args, **kwargs):
#             nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
#             def call_it():
#                 nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
#                 time_of_last_call = time()
#                 fn(*new_args, **new_kwargs)
#                 scheduled = False
#             time_since_last_call = time() - time_of_last_call
#             new_args, new_kwargs = args, kwargs
#             if not scheduled:
#                 scheduled = True
#                 new_wait = max(0, wait - time_since_last_call)
#                 timer = Timer(new_wait, call_it)
#                 timer.start()
#         return throttled
#     return decorator
