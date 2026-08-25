import numpy as np

from amp_finder.evaluation import (
    binary_classification_metrics,
    bootstrap_binary_metric_intervals,
    select_mcc_threshold,
)


def test_threshold_is_selected_without_test_data_and_metrics_are_consistent():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.20, 0.45, 0.55, 0.80, 0.95])
    threshold = select_mcc_threshold(labels, scores)
    metrics = binary_classification_metrics(labels, scores, threshold=threshold)
    assert 0 < threshold < 1
    assert metrics["mcc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 3}


def test_stratified_bootstrap_intervals_are_reproducible_and_bounded():
    labels = np.array([0] * 8 + [1] * 8)
    scores = np.array(
        [0.03, 0.10, 0.18, 0.24, 0.31, 0.43, 0.58, 0.65, 0.36, 0.49, 0.55, 0.61, 0.74, 0.82, 0.90, 0.97]
    )
    intervals = bootstrap_binary_metric_intervals(
        labels,
        scores,
        threshold=0.5,
        n_resamples=200,
        random_state=17,
    )

    assert intervals["n_resamples"] == 200
    assert intervals["confidence_level"] == 0.95
    assert "domain shift" in intervals["scope_caveat"]
    for interval in intervals["metrics"].values():
        assert interval["lower"] <= interval["estimate"] <= interval["upper"]
