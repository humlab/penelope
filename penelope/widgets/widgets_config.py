# from __future__ import print_function
import collections

import bokeh
import ipywidgets as widgets
import numpy as np

from penelope.utility import extend

AGGREGATES = { 'mean': np.mean, 'sum': np.sum, 'max': np.max, 'std': np.std }

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
    'classic'
]

output_formats = {
    'Plot vertical bar': 'plot_bar',
    'Plot horisontal bar': 'plot_barh',
    'Plot vertical bar, stacked': 'plot_bar_stacked',
    'Plot horisontal bar, stacked': 'plot_barh_stacked',
    'Plot line': 'plot_line',
    'Table': 'table',
    'Pivot': 'pivot'
}

KindOfChart = collections.namedtuple('KindOfChart', 'description name kind stacked horizontal')

CHART_TYPES = [
    KindOfChart(description='Area', name='plot_area', kind='area', stacked=False, horizontal=False),
    KindOfChart(description='Area (stacked)', name='plot_stacked_area', kind='area', stacked=True, horizontal=False),
    KindOfChart(description='Bar', name='plot_bar', kind='bar', stacked=False, horizontal=False),
    KindOfChart(description='Line', name='plot_line', kind='line', stacked=False, horizontal=False),
    KindOfChart(description='Bar (stacked)', name='plot_stacked_bar', kind='bar', stacked=True, horizontal=False),
    KindOfChart(description='Line (stacked)', name='plot_stacked_line', kind='line', stacked=True, horizontal=False),
    KindOfChart(description='Bar (horizontal)', name='plot_barh', kind='bar', stacked=False, horizontal=True),
    KindOfChart(description='Bar (horizontal, stacked)', name='plot_stacked_barh', kind='bar', stacked=True, horizontal=True),
    # KindOfChart(description='Scatter', name='plot_scatter', kind='scatter', stacked=False, horizontal=False),
    # KindOfChart(description='Histogram', name='plot_hist', kind='hist', stacked=False, horizontal=False),
    KindOfChart(description='Table', name='table', kind=None, stacked=False, horizontal=False),
    KindOfChart(description='Pivot', name='pivot', kind=None, stacked=False, horizontal=False)
]

CHART_TYPE_MAP = { x.name: x for x in CHART_TYPES }
CHART_TYPE_OPTIONS = { x.name: x.name for x in CHART_TYPES }
CHART_TYPE_NAME_OPTIONS = [ (x.description, x.name) for x in CHART_TYPES ]

FOX_STOPWORDS = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'because', 'become', 'becomes', 'became', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'herself', 'here', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest', 'next', 'no', 'non', 'not', 'nobody', 'noone', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'sees', 'seem', 'seemed', 'seeming', 'seems', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'uses', 'used', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours']


def kwargser(d):
    args = dict(d)
    if 'kwargs' in args:
        kwargs = args['kwargs']
        del args['kwargs']
        args.update(kwargs)
    return args


def toggle(description, value, **kwargs):  # pylint: disable=W0613
    return widgets.ToggleButton(**kwargser(locals()))


def toggles(description, options, value, **kwopts):  # pylint: disable=W0613
    return widgets.ToggleButtons(**kwargser(locals()))


def select_multiple(description, options, values, **kwopts):
    default_opts = dict(
        options=options,
        value=values,
        rows=4,
        description=description,
        disabled=False,
        layout=widgets.Layout(width='180px')
    )
    return widgets.SelectMultiple(**extend(default_opts, kwopts))


def dropdown(description, options, value, **kwargs):  # pylint: disable=unused-argument
    return widgets.Dropdown(**kwargser(locals()))


def slider(description, min, max, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return widgets.IntSlider(**kwargser(locals()))


def rangeslider(description, min, max, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return widgets.IntRangeSlider(**kwargser(locals()))


def sliderf(description, min, max, step, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return widgets.FloatSlider(**kwargser(locals()))


def progress(min, max, step, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return widgets.IntProgress(**kwargser(locals()))


def itext(min, max, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return widgets.BoundedIntText(**kwargser(locals()))


def wrap_id_text(dom_id, value=''):
    value = "<span class='{}'>{}</span>".format(dom_id, value) if dom_id is not None else value
    return value


def textblock(dom_id=None, value=''):
    return widgets.HTML(value=wrap_id_text(dom_id, value), placeholder='', description='')


def button(description):  # pylint: disable=unused-argument
    return widgets.Button(**kwargser(locals()))


def glyph_hover_js_code(element_id, id_name, text_name, glyph_name='glyph', glyph_data='glyph_data'):
    return """
        var indices = cb_data.index['1d'].indices;
        var current_id = -1;
        if (indices.length > 0) {
            var index = indices[0];
            var id = parseInt(""" + glyph_name + """.data.""" + id_name + """[index]);
            if (id !== current_id) {
                current_id = id;
                var text = """ + glyph_data + """.data.""" + text_name + """[id];
                $('.""" + element_id + """').html('ID ' + id.toString() + ': ' + text);
                #document.getElementsByClassName('""" + element_id + """')[0].innerText = 'ID ' + id.toString() + ': ' + text;
            }
    }
    """


def glyph_hover_callback2(glyph_source, glyph_id, text_ids, text, element_id):
    source = bokeh.models.ColumnDataSource(dict(text_id=text_ids, text=text))
    code = glyph_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
    callback = bokeh.models.CustomJS(args={'glyph': glyph_source, 'glyph_data': source}, code=code)
    return callback


def glyph_hover_callback(glyph_source, glyph_id, text_source, element_id):
    code = glyph_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
    callback = bokeh.models.CustomJS(args={'glyph': glyph_source, 'glyph_data': text_source}, code=code)
    return callback

def create_js_callback(axis, attribute, source):
    return bokeh.models.CustomJS(
        args=dict(source=source),
        code="""
        var data = source.data;
        var start = cb_obj.start;
        var end = cb_obj.end;
        data['""" + axis + """'] = [start + (end - start) / 2];
        data['""" + attribute + """'] = [end - start];
        source.change.emit();
    """
    )


def aggregate_function_widget(**kwopts):
    default_opts = dict(
        options=['mean', 'sum', 'std', 'min', 'max'],
        value='mean',
        description='Aggregate',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))


def years_widget(**kwopts):
    default_opts = dict(options=[], value=None, description='Year', layout=widgets.Layout(width='200px'))
    return widgets.Dropdown(**extend(default_opts, kwopts))


def plot_style_widget(**kwopts):
    default_opts = dict(
        options=[x for x in MATPLOTLIB_PLOT_STYLES if 'seaborn' in x],
        value='seaborn-pastel',
        description='Style:',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))


def increment_button(target_control, max_value, label='>>', increment=1):

    def f(_):
        target_control.value = (target_control.value + increment) % max_value

    return widgets.Button(description=label, callback=f)


def _get_field_values(documents, field, as_tuple=False, query=None):
    items = documents.query(query) if query is not None else documents
    unique_values = sorted(list(items[field].unique()))
    if as_tuple:
        unique_values = [(str(x).title(), x) for x in unique_values]
    return unique_values


def generate_field_filters(documents, opts):
    filters = []
    for opt in opts:  # if opt['type'] == 'multiselect':
        options = opt.get(
            'options', _get_field_values(documents, opt['field'], as_tuple=True, query=opt.get('query', None))
        )
        description = opt.get('description', '')
        rows = min(4, len(options))
        gf = extend(opt, widget=select_multiple(description, options, values=(), rows=rows))
        filters.append(gf)
    return filters
