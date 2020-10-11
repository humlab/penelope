from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import wordcloud


def plot_wordcloud(
    df: pd.DataFrame,
    token: str = 'token',
    weight: str = 'weight',
    figsize: Tuple[float, float] = (14, 14 / 1.618),
    **kwargs,
):
    """Plots a wordcloud using the `wordcloud` Python package

    Parameters
    ----------
    df : pd.DataFrame
        Tokens
    token : str, optional
        Name of token column, by default 'token'
    weight : str, optional
        Name of token weight column, by default 'weight'
    figsize : Tuple[float,float], optional
        Fig. size, by default (14, 14 / 1.618)
    """
    token_weights = dict({tuple(x) for x in df[[token, weight]].values})
    image = wordcloud.WordCloud(
        **kwargs,
    )
    image.fit_words(token_weights)
    plt.figure(figsize=figsize)  # , dpi=100)
    plt.imshow(image, interpolation='bilinear')
    plt.axis("off")
    plt.show()
