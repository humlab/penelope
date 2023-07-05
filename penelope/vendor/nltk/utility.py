# def _nltk_col_log_likelihood(count_a, count_b, count_ab, N):
#     """

#     https://www.nltk.org/_modules/nltk/tokenize/punkt.html

#     A function that will just compute log-likelihood estimate, in
#     the original paper it's decribed in algorithm 6 and 7.
#     This *should* be the original Dunning log-likelihood values,
#     unlike the previous log_l function where it used modified
#     Dunning log-likelihood values
#     """
#     import math

#     p = 1.0 * count_b / N
#     p1 = 1.0 * count_ab / count_a
#     p2 = 1.0 * (count_b - count_ab) / (N - count_a)

#     summand1 = count_ab * math.log(p) + (count_a - count_ab) * math.log(1.0 - p)

#     summand2 = (count_b - count_ab) * math.log(p) + (N - count_a - count_b + count_ab) * math.log(1.0 - p)

#     if count_a == count_ab:
#         summand3 = 0
#     else:
#         summand3 = count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)

#     if count_b == count_ab:
#         summand4 = 0
#     else:
#         summand4 = (count_b - count_ab) * math.log(p2) + (N - count_a - count_b + count_ab) * math.log(1.0 - p2)

#     likelihood = summand1 + summand2 - summand3 - summand4

#     return -2.0 * likelihood


# https://github.com/DrDub/icsisumm/blob/1cb583f86dddd65bfeec7bb9936c97561fd7811b/icsisumm-primary-sys34_v1/nltk/nltk-0.9.2/nltk/tokenize/punkt.py
