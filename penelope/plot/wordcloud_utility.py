from typing import Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    ...


# pylint: disable=unused-argument

try:
    import wordcloud

    def plot_wordcloud(
        df: pd.DataFrame,
        token: str = 'token',
        weight: str = 'weight',
        figsize: Tuple[float, float] = (14, 14 / 1.618),
        **kwargs,
    ):
        """Plots a wordcloud using the `wordcloud` Python package"""
        token_weights = dict({tuple(x) for x in df[[token, weight]].values})
        image = wordcloud.WordCloud(**kwargs)
        image.fit_words(token_weights)
        plt.figure(figsize=figsize)  # , dpi=100)
        plt.imshow(image, interpolation='bilinear')
        plt.axis("off")
        plt.show()

except ImportError:

    def plot_wordcloud(
        df: pd.DataFrame,
        token: str = 'token',
        weight: str = 'weight',
        figsize: Tuple[float, float] = (14, 14 / 1.618),
        **kwargs,
    ):
        print("wordcloud package not installed")
