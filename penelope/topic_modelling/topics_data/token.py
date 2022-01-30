from __future__ import annotations

from typing import Mapping, Protocol

import pandas as pd

# pylint: disable=no-member, useless-super-delegation


def get_topic_titles(
    topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100, id2token: dict[int, str] = None
) -> pd.Series:
    """Create string of `n_tokens` most probable words per topic."""

    weights: pd.DataFrame = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )

    if 'token' not in topic_token_weights.columns:

        if id2token is None:
            raise TypeError("get_topic_titles: either TTW must contain `token` or `id2token` must be supplied")

        fg = id2token.get
        topic_titles: pd.Series = (
            weights.sort_values('weight', ascending=False)
            .groupby('topic_id')
            .agg(token=('token_id', lambda x: ' '.join(map(fg, list(x)[:n_tokens]))))
        ).token

    else:
        topic_titles: pd.Series = (
            weights.sort_values('weight', ascending=False)
            .groupby('topic_id')
            .apply(lambda x: ' '.join(x.token[:n_tokens].str.title()))
        )

    return topic_titles


def get_topic_title(
    topic_token_weights: pd.DataFrame, topic_id: int, n_tokens: int = 100, id2token: dict[int, str] = None
) -> str:
    """Returns a string of `n_tokens` most probable words for topic `topic_id`"""
    return get_topic_titles(topic_token_weights, topic_id, n_tokens=n_tokens, id2token=id2token).iloc[0]


def get_topic_title2(
    topic_token_weights: pd.DataFrame, topic_id: int, n_tokens: int = 200, id2token: dict[int, str] = None
) -> str:
    """Returns a string of `n_tokens` most probable words for topic `topic_id` or message if not tokens."""
    tokens: str = (
        "Topics has no significant presence in any documents in the entire corpus"
        if len(topic_token_weights[topic_token_weights.topic_id == topic_id]) == 0
        else get_topic_title(topic_token_weights, topic_id, n_tokens=n_tokens, id2token=id2token)
    )

    return f'ID {topic_id}: {tokens}'


def _get_topic_top_tokens(
    ttw: pd.DataFrame, topic_id: int, n_tokens: int = 100, id2token: dict[int, str] = None
) -> pd.DataFrame:
    """Returns most probable tokens for given topic sorted by probability descending"""
    top_tokens: pd.DataFrame = ttw[(ttw.topic_id == topic_id)].sort_values('weight', ascending=False)[:n_tokens]
    top_tokens['token'] = top_tokens['token_id'].apply(id2token.get)
    return top_tokens


def top_topic_token_weights(topic_token_weights: pd.DataFrame, id2term: dict, n_top: int) -> pd.DataFrame:
    """Find top `n_top` tokens for each topic and their position."""
    data: pd.DataFrame = (
        topic_token_weights.groupby(['topic_id'])[['topic_id', 'token_id', 'weight']]
        .apply(lambda x: x.nlargest(n_top, columns=['weight']))
        .reset_index(drop=True)
    )
    data['token'] = data['token_id'].apply(lambda x: id2term[x])
    data['position'] = data.groupby('topic_id').cumcount() + 1
    return data.set_index('topic_id')


def filter_topic_tokens_overview(
    topic_tokens_overview: pd.DataFrame,
    *,
    search_text: str,
    n_top: int,
    truncate_tokens: bool = False,
    format_string: str = '<b style="color:green;font-size:14px">{}</b>',
) -> pd.DataFrame:
    """Filter out topics where `search` string is in `n_counts` words. Return data frame."""

    data: pd.DataFrame = pd.DataFrame(topic_tokens_overview)

    if search_text:
        top_tokens = data.tokens.apply(lambda x: x.split(' ')[:n_top]).str.join(' ')
        data = data[top_tokens.str.contains(search_text)]
        data['tokens'] = (top_tokens if truncate_tokens else data.tokens).apply(
            lambda x: x.replace(search_text, format_string.format(search_text))
        )

    return data


class IMixIn(Protocol):
    topic_token_weights: pd.DataFrame
    id2term: Mapping[int, str]


class TopicTokensMixIn:
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def top_topic_token_weights(self: IMixIn, n_top: int) -> pd.DataFrame:
        return top_topic_token_weights(self.topic_token_weights, id2term=self.id2term, n_top=n_top)

    def get_topic_titles(self: IMixIn, topic_id: int = None, n_tokens: int = 100) -> pd.Series:
        """Return strings of `n_tokens` most probable words per topic."""
        return get_topic_titles(self.topic_token_weights, topic_id, n_tokens=n_tokens, id2token=self.id2term)

    def get_topic_title(self: IMixIn, topic_id: int, n_tokens: int = 100) -> str:
        """Return string of `n_tokens` most probable words per topic"""
        return get_topic_title(self.topic_token_weights, topic_id, n_tokens=n_tokens, id2token=self.id2term)

    def get_topic_title2(self: IMixIn, topic_id: int, n_tokens: int = 100) -> str:
        """Return string of `n_tokens` most probable words per topic"""
        return get_topic_title2(self.topic_token_weights, topic_id, n_tokens=n_tokens, id2token=self.id2term)

    def get_topic_top_tokens(self: IMixIn, topic_id: int, n_tokens: int = 100) -> pd.DataFrame:
        """Return most probable tokens for given topic sorted by probability descending"""
        return _get_topic_top_tokens(
            self.topic_token_weights, topic_id=topic_id, n_tokens=n_tokens, id2token=self.id2term
        )

    def get_topic_tokens(self: IMixIn, topic_id: int) -> pd.DataFrame:
        topic_ttw: pd.DataFrame = self.topic_token_weights.loc[self.topic_token_weights.topic_id == topic_id]
        if 'token' not in topic_ttw:
            topic_ttw['token'] = topic_ttw['token_id'].apply(self.id2term.get)
        return topic_ttw
