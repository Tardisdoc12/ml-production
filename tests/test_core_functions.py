import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core_function import compute_metrics
import numpy as np

def test_compute_metrics():
    fake_pred = np.array([
        [0,0.7],
        [1,0.8],
        [0,0.71],
        [1,0.86],
        [0,0.73]
    ])
    fake_data = (fake_pred, [0, 1, 1, 0, 0])
    results = compute_metrics(fake_data)
    assert "f1" in results.keys()
    assert results["f1"] == 0.75

test_compute_metrics()