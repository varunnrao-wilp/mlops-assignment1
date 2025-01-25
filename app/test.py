import json

import pytest

from main import app


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint_returns_correct_status(client):
    """Test that the predict endpoint returns a 200 status code"""
    sample_input = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45,
        "total_sulfur_dioxide": 170,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
        "wine_type": 0
    }

    response = client.post('/predict',
                           data=json.dumps(sample_input),
                           content_type='application/json')

    assert response.status_code == 200


def test_predict_endpoint_returns_valid_prediction(client):
    """Test that the prediction is a valid numeric value"""
    sample_input = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45,
        "total_sulfur_dioxide": 170,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
        "wine_type": 0
    }

    response = client.post('/predict',
                           data=json.dumps(sample_input),
                           content_type='application/json')

    prediction = json.loads(response.data)['prediction']

    assert isinstance(prediction, (int, float))
    assert 0 <= prediction <= 10


def test_predict_endpoint_handles_invalid_input(client):
    """Test that the endpoint handles invalid input gracefully"""
    invalid_input = {
        "fixed_acidity": "not a number",
        "volatile_acidity": 0.27
    }

    response = client.post('/predict',
                           data=json.dumps(invalid_input),
                           content_type='application/json')

    assert response.status_code == 400
