import pytest
from run import create_app
import numpy as np
import requests


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict():
    test_data = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = requests.post("http://localhost:5000/predict", json=test_data)
    
    print("\n=== Prediction Results ===")
    print("Input Features:", test_data["features"])
    print("Predicted Class:", response.json()["prediction"])
    print("Probabilities:", response.json()["probability"])
    
    assert response.status_code == 200