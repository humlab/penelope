import os

from penelope.common.curve_fit import pchip_spline  # , rolling_average_smoother
from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends.displayers import TopTokensDisplayer

DEFAULT_SMOOTHERS = [pchip_spline]

folder = "/data/westac/shared/propaganda_w5_NNPM_PASSTHROUGH_LEMMA_STOPS"
tag = os.path.split(folder)[1]

corpus: VectorizedCorpus = VectorizedCorpus.load(folder=folder, tag=tag)

top_tokens = corpus.get_top_n_words(n=100000)
displayer: TopTokensDisplayer = TopTokensDisplayer()
displayer.setup()

indices = [x[1] for x in top_tokens]
smooth = False
print(len(indices))
plot_data = displayer.compile(corpus=corpus, indices=indices, smoothers=[DEFAULT_SMOOTHERS] if smooth else [])
# displayer.plot(plot_data=plot_data)

# displayer.display(
#     corpus=corpus,
#     indices=indices,
#     smooth=smooth,
# )
