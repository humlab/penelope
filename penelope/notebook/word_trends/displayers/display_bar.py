import itertools
import math

import bokeh

from . import data_compilers

NAME = "Bar"

# compile = data_compilers.compile_multiline_data  # pylint: disable=redefined-builtin
compile = data_compilers.compile_year_token_vector_data  # pylint: disable=redefined-builtin


def setup(container, **kwargs):  # pylint: disable=unused-argument
    pass


def plot(data, **_):

    years = [str(y) for y in data['year']]

    data['year'] = years

    tokens = [w for w in data.keys() if w != 'year']

    source = bokeh.models.ColumnDataSource(data=data)

    max_value = max([max(data[key]) for key in data if key != 'year']) + 0.005

    p = bokeh.plotting.figure(
        x_range=years, y_range=(0, max_value), plot_height=400, plot_width=1000, title="Word frequecy by year"
    )

    colors = itertools.islice(itertools.cycle(bokeh.palettes.d3['Category20'][20]), len(tokens))

    offset = -0.25
    v = []
    for token in tokens:
        w = p.vbar(
            x=bokeh.transform.dodge('year', offset, range=p.x_range),
            top=token,
            width=0.2,
            source=source,
            color=next(colors),
        )  # , legend_label=token)
        offset += 0.25
        v.append(w)

    p.x_range.range_padding = 0.04
    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    legend = bokeh.models.Legend(items=[(x, [v[i]]) for i, x in enumerate(tokens)])

    p.add_layout(legend, 'left')

    p.legend.location = "top_right"
    p.legend.orientation = "vertical"

    bokeh.io.show(p)
