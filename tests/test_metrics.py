from src.eval.metrics import average_precision_at_k, coverage, evaluate_recommendations, ndcg_at_k, recall_at_k


def test_recall_at_k_counts_hits_within_k():
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"c", "e", "z"}
    assert recall_at_k(recommended, relevant, k=5) == 2 / 3
    assert recall_at_k(recommended, relevant, k=2) == 0.0


def test_recall_at_k_empty_relevant_is_zero():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_average_precision_rewards_earlier_hits():
    relevant = {"a", "b"}
    ap_early = average_precision_at_k(["a", "b", "x"], relevant, k=3)
    ap_late = average_precision_at_k(["x", "a", "b"], relevant, k=3)
    assert ap_early > ap_late
    assert ap_early == 1.0


def test_ndcg_perfect_ranking_is_one():
    relevant = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], relevant, k=3) == 1.0


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_coverage_fraction_of_catalog_recommended():
    recs = [["a", "b"], ["b", "c"]]
    assert coverage(recs, catalog_size=6) == 3 / 6


def test_evaluate_recommendations_skips_users_without_ground_truth():
    recs = {"u1": ["a", "b"], "u2": ["x"]}
    truth = {"u1": {"a"}, "u2": set()}
    result = evaluate_recommendations(recs, truth, k=2, catalog_size=10)
    assert result["n_users"] == 1
    assert result["recall@k"] == 1.0
