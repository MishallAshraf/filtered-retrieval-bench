"""Retrieval metrics. Pure functions, so every expected value can be checked by hand."""
import math
import random


def recall_at_k(ranked, relevant, k=10):
    if not relevant:
        raise ValueError("a question with no relevant documents can't be scored")
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(ranked, relevant, k=10):
    if not relevant:
        raise ValueError("a question with no relevant documents can't be scored")
    relevant = set(relevant)
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(ranked[:k]) if doc in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def mrr_at_k(ranked, relevant, k=10):
    if not relevant:
        raise ValueError("a question with no relevant documents can't be scored")
    relevant = set(relevant)
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            return 1 / (i + 1)
    return 0.0


def recall_against_exact(found, exact):
    return len(set(found) & set(exact)) / max(len(exact), 1)


def rrf_scores(rankings, k=60):
    """Reciprocal Rank Fusion. Used to check the SQL version against hand-computed values."""
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    return scores


def bootstrap_ci(values, seed=42, samples=2000, level=0.95):
    """Mean with a percentile bootstrap confidence interval over questions."""
    if not values:
        raise ValueError("nothing to bootstrap")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    tail = (1 - level) / 2
    return sum(values) / n, means[int(tail * samples)], means[int((1 - tail) * samples) - 1]
