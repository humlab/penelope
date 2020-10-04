import itertools

import bokeh
from sklearn.decomposition import PCA


def compute(data, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data)
    return X_pca


def plot(X_pca, clusters=None):

    if clusters is not None:
        palette = itertools.cycle(bokeh.palettes.Colorblind8)
        cmap = {x: next(palette) for x in set(clusters)}
        colors = [cmap[x] for x in clusters]
    else:
        colors = 'navy'

    p = bokeh.plotting.figure(plot_width=600, plot_height=600)

    # p.legend(loc="best", shadow=False, scatterpoints=1)

    p.title.text = 'PCA decomposition'

    p.xaxis.axis_label, p.yaxis.axis_label = 'pca 0', 'pca 1'

    _ = p.scatter(X_pca[:, 0], X_pca[:, 1], size=6, fill_color=colors, color=colors, alpha=0.5)

    bokeh.plotting.show(p)

    return p


# def plot_clusters(x_corpus, metric, df_most_deviating):

#     tokens  = df_most_deviating.head(100)[metric+'_token'].tolist()
#     indices = [ x_corpus.token2id[w] for w in tokens ]
#     data    = x_corpus.data[:,indices].T

#     x_PCA   = compute(data)
#     p       = plot(x_PCA, clusters=df_clusters_at.label.values)
