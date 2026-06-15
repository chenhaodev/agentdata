"""Recombination engine — the user's core idea.

Cluster *similar subjects* (by shared meta tags, else by token overlap), then
combine their turns/events into new coherent multi-turn sessions — the LoCoMo
persona+event-graph method, generalized. Deterministic structure first (stable
clustering + interleave); optional LLM prose polish when a real provider exists.

This recombines existing real material into new samples without inventing facts,
so the output stays grounded while expanding coverage of multi_hop/temporal cases.
"""

from __future__ import annotations

from ..types import KIND_MESSAGES, DataItem
from .llm import LLMProvider, get_provider


def _tokens(item: DataItem) -> set[str]:
    return set(item.content_key().split())


def _cluster(items: list[DataItem], min_overlap: int = 2) -> list[list[DataItem]]:
    """Greedy, deterministic clustering: group by shared meta tag if present,
    else by token-overlap with the cluster seed. Order-stable for reproducibility."""
    # 1) tag-based grouping when tags exist
    tagged: dict[str, list[DataItem]] = {}
    untagged: list[DataItem] = []
    for it in items:
        tags = it.meta.get("tags") or []
        if tags:
            tagged.setdefault(str(sorted(tags)[0]), []).append(it)
        else:
            untagged.append(it)

    clusters = [grp for grp in tagged.values() if len(grp) >= 2]

    # 2) token-overlap grouping for the rest
    remaining = untagged + [it for grp in tagged.values() if len(grp) < 2 for it in grp]
    used = [False] * len(remaining)
    for i, seed in enumerate(remaining):
        if used[i]:
            continue
        group = [seed]
        used[i] = True
        seed_tok = _tokens(seed)
        for j in range(i + 1, len(remaining)):
            if used[j]:
                continue
            if len(seed_tok & _tokens(remaining[j])) >= min_overlap:
                group.append(remaining[j])
                used[j] = True
        if len(group) >= 2:
            clusters.append(group)
    return clusters


def recombine(items: list[DataItem], provider: LLMProvider | None = None,
              max_per_cluster: int = 4, limit: int = 0,
              multi_agent: bool = False) -> list[DataItem]:
    """Produce recombined multi-turn sessions, one per qualifying cluster.

    `multi_agent=True` turns each cluster into a **role-conditioned multi-agent
    transcript**: every recombined subject's assistant turns are relabeled to a
    distinct agent role (`agent1`, `agent2`, ...), so the output is a multi-party
    dialogue suitable for training/evaluating multi-agent systems (the chat/sharegpt
    emitters preserve these roles)."""
    provider = provider or get_provider("mock")
    out: list[DataItem] = []
    for cluster in _cluster(items):
        members = cluster[:max_per_cluster]
        merged: list[dict[str, str]] = []
        sources: list[str] = []
        for i, m in enumerate(members):
            turns = m.messages if m.kind == KIND_MESSAGES else _as_turn(m)
            for msg in turns:
                if not msg.get("content", "").strip():
                    continue
                if multi_agent and msg.get("role") == "assistant":
                    msg = {**msg, "role": f"agent{i + 1}"}  # distinct agent per subject
                merged.append(msg)
            src = m.meta.get("source")
            if src and src not in sources:
                sources.append(src)
        if len(merged) < 2:
            continue
        item = DataItem(
            kind=KIND_MESSAGES, messages=merged,
            meta={"source": "recombine", "synthetic": True, "gen": "recombine",
                  "recombined_from": sources, "n_subjects": len(members),
                  "multi_agent": multi_agent},
        )
        out.append(item)
        if limit and len(out) >= limit:
            break
    return out


def _as_turn(item: DataItem) -> list[dict[str, str]]:
    if item.kind == "qa":
        return [{"role": "user", "content": item.question},
                {"role": "assistant", "content": item.answer}]
    return [{"role": "user", "content": item.text}]
