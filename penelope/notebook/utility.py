import os
import types
from typing import Any

import bokeh.plotting
import ipyfilechooser
import ipywidgets as widgets
import pandas as pd
import yaml
from IPython.display import Javascript
from IPython.display import display as ipython_display
from loguru import logger

# pylint: disable=too-many-ancestors

CLEAR_OUTPUT = True


def create_js_download(df: pd.DataFrame, filename='results.csv', **to_csv_opts) -> Javascript:

    if df is None or len(df) == 0:
        return None

    csv_text = df.to_csv(**to_csv_opts).replace('\n', '\\n').replace('\r', '').replace("'", "\'")

    js_download = """
        var csv = '%s';
        var filename = '%s';
        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        if (navigator.msSaveBlob) { // IE 10+
            navigator.msSaveBlob(blob, filename);
        } else {
            var link = document.createElement("a");
            if (link.download !== undefined) { // HTML5 check
                var url = URL.createObjectURL(blob);
                link.setAttribute("href", url);
                link.setAttribute("download", filename);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
    """ % (
        csv_text,
        filename,
    )

    return Javascript(js_download)


class OutputsTabExt(widgets.Tab):
    def __init__(self, names, **kwargs):
        super().__init__(**kwargs)
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

    def selective_plot(self, what: Any):
        that = what() if callable(what) else what
        if isinstance(that, bokeh.plotting.Figure):
            bokeh.plotting.show(that)
        else:
            ipython_display(that)

    def display_content(self, i: int, what: Any, clear: bool = False, plot=True):

        try:
            if clear:

                self.children[i].clear_output()
                self.loaded[i] = False

            with self.children[i]:

                if not self.loaded[i]:

                    if plot:

                        self.selective_plot(what)

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


def dummy_context():
    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
            pass

    return DummyContext()


def default_done_callback(*_, **__):
    print("Vectorization done!")


def shorten_path_with_ellipsis(path: str, max_length: int):
    if len(path) > max_length:
        path, filename = os.path.split(path)
        path = f"{path[:max(0, max_length-len(filename))]}.../{filename}"
    return path


def shorten_filechooser_label(fc: ipyfilechooser.FileChooser, max_length: int):
    try:
        template = getattr(fc, '_LBL_TEMPLATE')
        if not template or not hasattr(template, 'format'):
            return
        fake = types.SimpleNamespace(
            format=lambda p, c: template.format(
                shorten_path_with_ellipsis(p, max_length),
                'green' if os.path.exists(p) else 'black',
            )
        )
        setattr(fc, '_LBL_TEMPLATE', fake)
        if len(fc.selected) > max_length:
            getattr(fc, '_label').value = fake.format(None, fc.selected, 'green')
    except:  # pylint: disable=bare-except
        pass


class FileChooserExt(ipyfilechooser.FileChooser):

    label_max_length = 50

    _LBL_TEMPLATE = types.SimpleNamespace(
        format=lambda p, c: super()._LBL_TEMPLATE.format(
            shorten_path_with_ellipsis(p, FileChooserExt.label_max_length),
            'green' if os.path.exists(p) else 'black',
        )
    )


class FileChooserExt2(ipyfilechooser.FileChooser):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        path=os.getcwd(),
        filename='',
        title='',
        select_desc='Select',
        change_desc='Change',
        show_hidden=False,
        select_default=False,
        use_dir_icons=False,
        show_only_dirs=False,
        filter_pattern=None,
        label_max_length=50,
        **kwargs,
    ):
        self._LBL_TEMPLATE = types.SimpleNamespace(
            format=lambda p, c: ipyfilechooser.FileChooser._LBL_TEMPLATE.format(
                shorten_path_with_ellipsis(p, label_max_length), c
            )
        )
        """ Strip dirs from filename (raises error otherwise) """
        filename = os.path.basename(filename or '')
        super().__init__(
            path=path,
            filename=filename,
            title=title,
            select_desc=select_desc,
            change_desc=change_desc,
            show_hidden=show_hidden,
            select_default=select_default,
            use_dir_icons=use_dir_icons,
            show_only_dirs=show_only_dirs,
            filter_pattern=filter_pattern,
            **kwargs,
        )
