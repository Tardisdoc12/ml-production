# tests/test_api.py
import requests


def test_predict_endpoint():
    payload = {"features": ["you are so ugly and fat. go suicide!"]}
    resp = requests.post("http://localhost:8000/predict", json=payload)
    assert resp.status_code == 200
    assert "prediction" in resp.json()
