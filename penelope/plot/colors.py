from itertools import cycle

from bokeh.palettes import Category20

kelly_colors = [
    '#d11141',
    '#00b159',
    '#00aedb',
    '#f37735',
    '#edc951',
    '#eb6841',
    '#ffc425',
    '#cc2a36',
    '#4f372d',
    '#00a0b0',
    # Kellys colors
    '#F2F3F4',
    '#222222',
    '#F3C300',
    '#875692',
    '#F38400',
    '#A1CAF1',
    '#BE0032',
    '#C2B280',
    '#848482',
    '#008856',
    '#E68FAC',
    '#0067A5',
    '#F99379',
    '#604E97',
    '#F6A600',
    '#B3446C',
    '#DCD300',
    '#882D17',
    '#8DB600',
    '#654522',
    '#E25822',
    '#2B3D26',
]


def get_color_palette():
    colors = cycle(kelly_colors + list(Category20[20]))

    return colors
