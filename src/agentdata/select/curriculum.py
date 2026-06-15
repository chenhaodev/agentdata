"""Hard-biased stratified selection + easy→hard curriculum ordering.

Port of medqa's `hard_biased_select` with the same quartile budget
(Q1..Q4 = 10/20/30/40%), implemented in pure Python (seeded `random`) so it needs
no numpy and is deterministic. Returns items sorted ascending by difficulty
(easy-first) for curriculum training.
"""

from __future__ import annotations

import random

from ..types import DataItem
from .score import score_difficulty

_QUARTILE_WEIGHTS = (0.10, 0.20, 0.30, 0.40)  # Q1(easiest) .. Q4(hardest)


def curriculum_select(items: list[DataItem], n_target: int = 0, seed: int = 42) -> list[DataItem]:
    """Select (hard-biased) and sort (easy→hard).

    `n_target <= 0` or >= len(items) keeps everything but still curriculum-sorts.
    """
    if not items:
        return []
    # score once (regex + message build is not free; the pool can be huge), then
    # reuse the cached score everywhere — sort, quartile split, final ordering.
    paired = sorted(((score_difficulty(it), it) for it in items), key=lambda p: p[0])
    if n_target <= 0 or n_target >= len(paired):
        return [it for _s, it in paired]

    n = len(paired)
    bounds = [0, n // 4, n // 2, 3 * n // 4, n]
    quartiles = [paired[bounds[i]:bounds[i + 1]] for i in range(4)]

    rng = random.Random(seed)
    chosen: list[tuple[float, DataItem]] = []
    for q_items, w in zip(quartiles, _QUARTILE_WEIGHTS):
        if not q_items:
            continue
        budget = min(max(1, round(n_target * w)), len(q_items))
        chosen.extend(rng.sample(q_items, budget))

    chosen.sort(key=lambda p: p[0])  # easy→hard curriculum order
    return [it for _s, it in chosen]
