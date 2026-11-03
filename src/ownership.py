"""Assign FiQA documents to simulated users so human relevance labels can be scored.

With random ownership a 1% user owns a relevant document for about 3% of questions, too
few to score. So questions are split between users, each user receives the relevant
documents for their questions, and the rest of their share is filled at random.
"""
import random

SEED = 42

# Share of the corpus each test user owns in the random assignment. The remainder belongs to "other".
SHARES = {"u_50pct": 0.50, "u_10pct": 0.10, "u_5pct": 0.05, "u_1pct": 0.01, "u_0_1pct": 0.001}

# Share of the corpus, and questions per split, for each user. The 0.1% user owns about
# 57 documents, too few to hold enough questions, so that user is measured only against
# exact scan.
LABELLED_USERS = {
    "u_50pct": {"share": 0.50, "dev": 250, "test": 200},
    "u_10pct": {"share": 0.10, "dev": 130, "test": 178},
    "u_5pct": {"share": 0.05, "dev": 60, "test": 150},
    "u_1pct": {"share": 0.01, "dev": 60, "test": 120},
}


def assign_owners(n):
    """Random ownership, used to measure results against an exact scan."""
    ids = list(range(n))
    random.Random(SEED).shuffle(ids)
    owners = ["other"] * n
    start = 0
    for user, share in SHARES.items():
        size = int(n * share)
        for i in ids[start:start + size]:
            owners[i] = user
        start += size
    return owners


def assign_labelled(doc_count, questions, users=None, seed=SEED):
    """Return (owners, assignments).

    questions maps split -> {question_id: [document index, ...]}. owners holds one user
    per document. assignments maps user -> split -> {question_id: [document index, ...]}.
    A question is dropped when another user already owns one of its relevant documents,
    or when it would overfill the user's share.
    """
    users = users or LABELLED_USERS
    rng = random.Random(seed)
    owners = ["other"] * doc_count
    quota = {user: int(doc_count * spec["share"]) for user, spec in users.items()}
    assignments = {user: {split: {} for split in questions} for user in users}

    for split, by_question in questions.items():
        ids = sorted(by_question)
        rng.shuffle(ids)
        start = 0
        for user, spec in users.items():
            for question in ids[start:start + spec[split]]:
                docs = by_question[question]
                if not docs or any(owners[d] not in ("other", user) for d in docs):
                    continue
                new = [d for d in docs if owners[d] == "other"]
                if len(new) > quota[user]:
                    continue
                for d in new:
                    owners[d] = user
                quota[user] -= len(new)
                assignments[user][split][question] = list(docs)
            start += spec[split]

    free = [i for i, owner in enumerate(owners) if owner == "other"]
    rng.shuffle(free)
    start = 0
    for user in users:
        for i in free[start:start + quota[user]]:
            owners[i] = user
        start += quota[user]
    return owners, assignments
