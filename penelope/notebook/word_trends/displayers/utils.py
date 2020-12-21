from typing import List, Tuple


def get_year_category_ticks(categories: List[int], n_tick: int = 5) -> List[int]:
    """Gets ticks every n_tick years if category is year
    Returns all cateories if all values are either, lustrum and decade"""

    if all([x % 5 in (0, 5) for x in categories]):
        return categories

    return list(range(low_bound(categories, n_tick), high_bound(categories, n_tick) + 1, n_tick))


def high_bound(categories: List[int], n_tick: int) -> Tuple[int, int]:
    return (lambda x: x if x % n_tick == 0 else x + (n_tick - x % n_tick))(int(max(categories)))


def low_bound(categories: List[int], n_tick: int) -> int:
    return (lambda x: x - (x % n_tick))(int(min(categories)))
