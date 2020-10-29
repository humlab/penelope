
SIMPLE_CORPUS_ABCDE_5DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c']),
    ('tran_2019_02_test.txt', ['a', 'a', 'b', 'd']),
    ('tran_2019_03_test.txt', ['a', 'e', 'e', 'b']),
    ('tran_2020_01_test.txt', ['c', 'c', 'd', 'a']),
    ('tran_2020_02_test.txt', ['a', 'b', 'b', 'e']),
]

SIMPLE_CORPUS_ABCDEFG_7DOCS = [
    ('rand_1991_1.txt', ['b', 'd', 'a', 'c', 'e', 'b', 'a', 'd', 'b']),
    ('rand_1992_2.txt', ['b', 'f', 'e', 'e', 'f', 'e', 'a', 'a', 'b']),
    ('rand_1992_3.txt', ['a', 'e', 'f', 'b', 'e', 'a', 'b', 'f']),
    ('rand_1992_4.txt', ['e', 'a', 'a', 'b', 'g', 'f', 'g', 'b', 'c']),
    ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
    ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
    ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
]

SIMPLE_CORPUS_ABCDEFG_3DOCS = [
    ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
    ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
    ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
]

SAMPLE_WINDOW_STREAM = [
    ['rand_1991_1.txt', 0, ['*', '*', 'b', 'd', 'a']],
    ['rand_1991_1.txt', 1, ['c', 'e', 'b', 'a', 'd']],
    ['rand_1991_1.txt', 2, ['a', 'd', 'b', '*', '*']],
    ['rand_1992_2.txt', 0, ['*', '*', 'b', 'f', 'e']],
    ['rand_1992_2.txt', 1, ['a', 'a', 'b', '*', '*']],
    ['rand_1992_3.txt', 0, ['e', 'f', 'b', 'e', 'a']],
    ['rand_1992_3.txt', 1, ['e', 'a', 'b', 'f', '*']],
    ['rand_1992_4.txt', 0, ['a', 'a', 'b', 'g', 'f']],
    ['rand_1992_4.txt', 1, ['f', 'g', 'b', 'c', '*']],
    ['rand_1991_5.txt', 0, ['*', 'c', 'b', 'c', 'e']],
    ['rand_1991_6.txt', 0, ['*', 'f', 'b', 'g', 'a']],
]
