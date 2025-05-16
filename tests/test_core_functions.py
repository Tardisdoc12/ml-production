from src.core_function import compute_metrics


def test_compute_metrics():
    fake_data = ([0, 1, 0, 2, 3], [0, 2, 0, 1, 3])
    results = compute_metrics(fake_data)
    print(results)
    assert "f1" in results.keys()
    assert results["f1"] == 3 / 5
