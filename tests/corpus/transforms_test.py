from typing import Callable, Iterable

import pytest

from penelope.corpus import transform as tr
from penelope.utility.utils import CommaStr
from penelope.vendor.nltk import STOPWORDS_CACHE, extended_stopwords, get_stopwords, load_stopwords

# pylint: disable=protected-access


def test_transform_smoke_test():
    assert list(map(str.lower, ['RÄK', 'SMÖR', 'GÅS'])) == ['räk', 'smör', 'gås']

    assert 'a\nb\nc' == '\n'.join(map(str.strip, 'a\n  b\n  \tc'.split('\n')))

    assert tr.TextTransformRegistry._items != tr.TokensTransformRegistry._items
    assert tr.TextTransformRegistry._aliases != tr.TokensTransformRegistry._aliases

    assert all('_' not in k for k in tr.TextTransformRegistry._items)
    assert all(',' not in k for k in tr.TextTransformRegistry._items)

    assert all('_' not in k for k in tr.TextTransformRegistry._aliases)
    assert all(',' not in k for k in tr.TextTransformRegistry._aliases)


def test_normalize_characters():
    text = "räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗RÄKSMÖRGÅS"
    normalized_text = tr.normalize_characters(text)
    assert normalized_text == 'räksmörgås----------++//~~~~~~~\'\'\'\'\'\'\'\'\'""""`′′′′′′RÄKSMÖRGÅS'

    text = "räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗RÄKSMÖRGÅS"
    normalized_text = tr.normalize_characters(text, groups="double_quotes,tildes")
    assert normalized_text == 'räksmörgås‐‑⁃‒–—―−－⁻＋⁺⁄∕~~~~~~~’՚Ꞌꞌ＇‘’‚‛""""´″‴‵‶‷⁗RÄKSMÖRGÅS'


def test_resolve_fx_callable():
    function = lambda x: x * 2
    result = tr.TransformRegistry.resolve_fx(function)
    assert callable(result)
    assert result(2) == 4


def test_resolve_fx_string_no_args():
    overrides = {'len': len}
    result = tr.TransformRegistry.resolve_fx('len', overrides)
    assert callable(result)
    assert result('test') == 4


try:
    from distutils.util import strtobool # type: ignore=deprecated-module
except ImportError:
    from setuptools.distutils.util import strtobool  # type: ignore=import-error


def test_resolve_fx_string_with_args():
    overrides = {'apify': lambda t, b: 'apa' if strtobool(b) else t}
    fx = tr.TransformRegistry.resolve_fx('apify?False', overrides)
    assert callable(fx)
    assert fx("hej") == "hej"
    fx = tr.TransformRegistry.resolve_fx('apify?True', overrides)
    assert callable(fx)
    assert fx("hej") == "apa"


def test_resolve_fx_string_with_kwargs():
    overrides = {'apify': lambda t, is_apa, name: name if strtobool(is_apa) else t}
    result = tr.TransformRegistry.resolve_fx('apify?is_apa=True&name=apa', overrides)
    assert callable(result)
    assert result('hej') == 'apa'
    result = tr.TransformRegistry.resolve_fx('apify?is_apa=False&name=apa', overrides)
    assert callable(result)
    assert result('hej') == 'hej'


def test_resolve_fx_string_with_list_args():
    overrides = {'apify': lambda t, is_apa, name: name if strtobool(is_apa) else t}
    result = tr.TransformRegistry.resolve_fx('apify?[True,apa]', overrides)
    assert callable(result)
    assert result('hej') == 'apa'
    result = tr.TransformRegistry.resolve_fx('apify?[False,apa]', overrides)
    assert callable(result)
    assert result('olle') == 'olle'


def test_resolve_fx_invalid():
    with pytest.raises(ValueError):
        tr.TransformRegistry.resolve_fx(123)


def test_resolve_fx_string_not_callable():
    with pytest.raises(ValueError):
        tr.TransformRegistry.resolve_fx('unknown')


def test_transformers():
    assert ['A', 'B', 'C'] == list(tr.to_upper(['a', 'b', 'c']))
    assert ['A', 'B', 'C'] == list(tr.TokensTransformRegistry.get("to-upper")(['a', 'b', 'c']))

    assert ['a', 'b', 'c'] == list(tr.to_lower(['A', 'B', 'C']))
    assert ['a', 'b', 'c'] == list(tr.TokensTransformRegistry.get("to-lower")(['A', 'b', 'C']))

    assert ['a', 'b', 'c'] == list(tr.remove_symbols(['a', 'b', '#', '_', '%', 'c']))
    assert ['a', 'b', 'c'] == list(tr.TokensTransformRegistry.get("remove-symbols")(['a', 'b', '#', '_', '%', 'c']))

    assert ['a', 'c'] == list(tr.remove_numerals(['a', '1', 'c']))
    assert ['a', 'c'] == list(tr.TokensTransformRegistry.get("remove-numerals")(['a', '1', 'c']))

    assert ['a', 'b', 'c'] == list(tr.only_alphabetic(['a', 'b', '#', '_', '%', 'c']))
    assert ['a', 'c'] == list(tr.TokensTransformRegistry.get("only-alphabetic")(['a', '1', 'c']))

    assert ['a', 'b', 'c'] == list(tr.has_alphabetic(['a', 'b', '#', '_', '%', 'c']))
    assert ['a', 'c'] == list(tr.TokensTransformRegistry.get("has-alphabetic")(['a', '1', 'c']))

    assert ['a', 'b', 'c', '2c'] == list(tr.only_any_alphanumeric(['a', 'b', 'c', '#', '_', '%', '2c']))
    assert ['a', '1', 'c'] == list(tr.TokensTransformRegistry.get("only-any-alphanumeric")(['a', '1', 'c']))

    assert ['a', 'b', 'c'] == list(tr.remove_empty(['a', 'b', 'c']))
    assert ['a', 'c'] == list(tr.TokensTransformRegistry.get("remove-empty")(['a', '', 'c']))

    assert 'a\nb\nc' == tr.dedent('a\n  b\n  \tc')
    assert 'a\nb\nc' == tr.TextTransformRegistry.get("dedent")('a\n  b\n  \tc')

    assert 'Mahler' == tr.strip_accents('Mähler')
    assert 'Mahler' == tr.TextTransformRegistry.get("strip-accents")('Mähler')

    assert 'Mähler' == tr.space_after_period_uppercase('Mähler')
    assert '. Mähler' == tr.space_after_period_uppercase('.Mähler')
    assert 'stenen. Mähler' == tr.space_after_period_uppercase('stenen.Mähler')
    assert 'stenen.mähler' == tr.space_after_period_uppercase('stenen.mähler')
    assert 'Mahler' == tr.TextTransformRegistry.get("strip-accents")('Mähler')
    assert 'stenen. Mähler' == tr.TextTransformRegistry.get('fix-space-after-sentence')('stenen.Mähler')


def test_reduced_transformers():

    assert 'stenen. mähler' == tr.TextTransformRegistry.getfx('fix-space-after-sentence,lowercase')('stenen.Mähler')
    assert 'stenen. mähler' == tr.TextTransformRegistry.getfx('fix-space-after-sentence', 'lowercase')('stenen.Mähler')
    assert 'apa' == tr.TextTransformRegistry.getfx(
        'fix-space-after-sentence', 'lowercase', overrides={'fix-space-after-sentence': lambda _: 'APA'}
    )('stenen.Mähler')
    assert "stenen. bergsgeten\n." == tr.TextTransformRegistry.getfx(
        'dehyphen,fix-space-after-sentence,normalize-whitespace,lowercase'
    )(
        """stenen.Bergs-
        geten."""
    )


def test_get_transformers():
    assert callable(tr.TextTransformRegistry.get("strip-accents"))
    assert callable(tr.TextTransformRegistry.get("  dedent   "))

    assert callable(tr.TokensTransformRegistry.get("remove-empty"))

    with pytest.raises(ValueError):
        tr.TextTransformRegistry.get("strip-accents,remove-empty")

    assert len(tr.TokensTransformRegistry.gets("remove-empty,to-lower,to-upper")) == 3

    _fx: Callable[[Iterable[str]], Iterable[str]] = tr.TokensTransformRegistry.getfx(
        "remove-empty,to-upper,has-alphabetic"
    )

    assert callable(_fx)
    assert ['A', 'B', 'C'] == list(_fx(['a', 'b', '', '$', '1', 'c']))


def test_tokenize():
    _fx: Callable[[str], Iterable[str]] = tr.TextTransformRegistry.getfx("sparv-tokenize")
    assert callable(_fx)
    assert ['A', 'B', 'C', '.'] == list(_fx('A B C.'))


def test_min_chars():
    _fx: Callable[[str], Iterable[str]] = tr.TokensTransformRegistry.getfx("min-chars?chars=2")
    assert callable(_fx)
    assert ['AB', 'CD', 'EF'] == list(_fx(['A', 'AB', 'CD', 'Y', 'EF', 'G']))


def test_minmax_chars():
    _fx: Callable[[str], Iterable[str]] = tr.TokensTransformRegistry.getfx("min-chars?2")
    assert callable(_fx)
    assert ['AB', 'CD', 'EF'] == list(_fx(['A', 'AB', 'CD', 'Y', 'EF', 'G']))

    _fx: Callable[[str], Iterable[str]] = tr.TokensTransformRegistry.getfx("max-chars?1")
    assert callable(_fx)
    assert ['A', 'Y', 'G'] == list(_fx(['A', 'AB', 'CD', 'Y', 'EF', 'G']))

    _fx: Callable[[str], Iterable[str]] = tr.TokensTransformRegistry.getfx("max-chars?chars=1")
    assert ['A', 'Y', 'G'] == list(_fx(['A', 'AB', 'CD', 'Y', 'EF', 'G']))


class HolaBandola:
    def __init__(self):
        self.registry: tr.TextTransformRegistry = tr.TextTransformRegistry

    def __getattr__(self, name):
        return self.registry.get(name.replace('_', '-'))


def test_hola_bandola():
    item = HolaBandola()

    assert 'Mahler' == item.strip_accents('Mähler')


def test_transform_load_stopwords():
    stopwords = load_stopwords("swedish")

    assert isinstance(stopwords, set)

    assert 'och' in stopwords

    stopwords_plus = load_stopwords("swedish", {"apa", "paj"})

    assert stopwords_plus.difference(stopwords.union({"apa", "paj"})) == set()


def test_remove_stopwords():
    stopwords = load_stopwords("swedish")

    extra_stopwords = extended_stopwords("swedish")

    assert len(extra_stopwords) == len(stopwords)
    assert len(STOPWORDS_CACHE) > 0

    _fx: Callable[[str], Iterable[str]] = tr.TokensTransformRegistry.getfx("remove-stopwords?swedish")
    assert callable(_fx)
    assert ['springa', 'fönster'] == list(_fx(['springa', 'och', 'fönster', 'hon', 'det']))

    _fx: Callable[[str], Iterable[str]] = tr.TokensTransformRegistry.getfx("remove-stopwords?english")
    assert callable(_fx)
    assert ['run', 'window'] == list(_fx(['run', 'and', 'window', 'she', 'it']))


def test_get_stopwords():
    words = get_stopwords("swedish")
    assert len(words) > 0


def cfx(
    transforms: str | CommaStr | dict[str, bool] = None,
    extras: list[Callable] = None,
    extra_stopwords: list[str] = None,
) -> Callable[[list[str]], list[str]]:
    opts = tr.TokensTransformOpts(transforms=transforms, extras=extras, extra_stopwords=extra_stopwords)
    return lambda tokens: list(opts.getfx()(tokens))


@pytest.mark.parametrize(
    'transforms,tokens,expected_tokens',
    [
        ({'to-lower': True}, ['A', 'B', 'C'], ['a', 'b', 'c']),
        ('to-lower', ['A', 'B', 'C'], ['a', 'b', 'c']),
        ('to-lower,to-upper', ['A', 'B', 'C'], ['A', 'B', 'C']),
        ('to-upper,to-lower', ['A', 'B', 'C'], ['a', 'b', 'c']),
        ('to-upper,min-chars?2', ['A', 'B', 'C'], []),
        ('to-upper,min-chars?2', ['A', 'AB', 'BC', 'CD', 'E'], ['AB', 'BC', 'CD']),
        ('to-upper,min-chars?2,max-chars?2', ['A', 'AB', 'BC', 'XYZ', 'CD', 'E', 'ABC'], ['AB', 'BC', 'CD']),
    ],
)
def test_transform_opts(transforms: str, tokens, expected_tokens):
    opts = tr.TokensTransformOpts(transforms=transforms, extras=None, extra_stopwords=None)
    results = list(opts.getfx()(tokens))
    assert results == expected_tokens


def test_transform_empty_preprocessor():
    opts = tr.TokensTransformOpts(transforms="")
    assert list(opts.getfx()(['A', 'B', 'C'])) == ['A', 'B', 'C']


def test_transform_opts_extras():
    def to_apa(tokens: list[str]) -> list[str]:
        return ['APA' for _ in tokens]

    opts = tr.TokensTransformOpts(transforms="", extras=[to_apa], extra_stopwords=None)
    results = list(opts.getfx()(['A', 'B', 'C']))
    assert results == ['APA', 'APA', 'APA']


def test_transform_opts_extras_stopwords():
    assert cfx('', extra_stopwords=['A', 'B'])(['A', 'B', 'C']) == ['C']


def test_transform_property_descriptor():
    assert tr.TokensTransformOpts({'to-lower': True}).to_lower
    assert tr.TokensTransformOpts({'to_lower': True}).to_lower
    assert tr.TokensTransformOpts('to-lower').to_lower
    assert tr.TokensTransformOpts('to_lower').to_lower

    assert tr.TokensTransformOpts('to_lower?True').to_lower
    assert tr.TokensTransformOpts('to_lower?False').to_lower is False

    assert tr.TokensTransformOpts('max-chars?1').max_len == 1
    assert tr.TokensTransformOpts('remove-stopwords?swedish').remove_stopwords == 'swedish'
