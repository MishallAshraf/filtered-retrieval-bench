from ownership import assign_labelled

USERS = {
    "a": {"share": 0.2, "test": 2},
    "b": {"share": 0.1, "test": 2},
}
QUESTIONS = {"test": {"q1": [0, 1], "q2": [2], "q3": [3], "q4": [4]}}


def test_each_user_owns_every_relevant_document_for_their_questions():
    owners, assignments = assign_labelled(100, QUESTIONS, USERS)
    for user, splits in assignments.items():
        for docs in splits["test"].values():
            assert all(owners[d] == user for d in docs)


def test_no_question_is_shared_and_shares_are_exact():
    owners, assignments = assign_labelled(100, QUESTIONS, USERS)
    assert not set(assignments["a"]["test"]) & set(assignments["b"]["test"])
    assert owners.count("a") == 20
    assert owners.count("b") == 10


def test_question_is_dropped_when_another_user_owns_its_document():
    questions = {"test": {"q1": [0], "q2": [0], "q3": [1], "q4": [0]}}
    owners, assignments = assign_labelled(100, questions, USERS)
    kept = list(assignments["a"]["test"]) + list(assignments["b"]["test"])
    holders = {user for user, splits in assignments.items()
               for docs in splits["test"].values() if 0 in docs}
    assert len(holders) <= 1
    assert all(owners[d] == user for user, splits in assignments.items()
               for docs in splits["test"].values() for d in docs)
    assert len(kept) < 4


def test_question_that_overfills_a_small_share_is_dropped():
    users = {"small": {"share": 0.01, "test": 1}}
    owners, assignments = assign_labelled(100, {"test": {"big": [0, 1, 2]}}, users)
    assert assignments["small"]["test"] == {}
    assert owners.count("small") == 1


def test_same_seed_repeats_and_another_seed_differs():
    first = assign_labelled(1000, QUESTIONS, USERS, seed=1)
    assert first == assign_labelled(1000, QUESTIONS, USERS, seed=1)
    assert first[0] != assign_labelled(1000, QUESTIONS, USERS, seed=2)[0]
