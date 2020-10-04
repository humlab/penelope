import matplotlib.pyplot as plt
import wordcloud


def plot_wordcloud(df_data, token='token', weight='weight', **args):
    token_weights = dict({tuple(x) for x in df_data[[token, weight]].values})
    image = wordcloud.WordCloud(
        **args,
    )
    image.fit_words(token_weights)
    plt.figure(figsize=(12, 12))  # , dpi=100)
    plt.imshow(image, interpolation='bilinear')
    plt.axis("off")
    # plt.set_facecolor('w')
    # plt.tight_layout()
    plt.show()


# def plot_wordcloud(df_data, token='token', weight='weight', figsize=(12, 12), dpi=100, **args):
#     token_weights = dict({ tuple(x) for x in df_data[[token, weight]].values })
#     image = wordcloud.WordCloud(**args,)
#     image.fit_words(token_weights)
#     plt.figure(figsize=figsize, dpi=dpi)
#     plt.imshow(image, interpolation='bilinear')
#     plt.axis("off")
#     # plt.set_facecolor('w')
#     # plt.tight_layout()
#     plt.show()
