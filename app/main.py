import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'winequality.joblib')
model = joblib.load(MODEL_PATH)


@app.route('/ping', methods=['GET'])
def pong():
    return jsonify({"message": "pong"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_features = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid',
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH',
            'sulphates', 'alcohol', "wine_type"
        ]

        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing required features"}), 400

        input_features = np.array([
            data['fixed_acidity'],
            data['volatile_acidity'],
            data['citric_acid'],
            data['residual_sugar'],
            data['chlorides'],
            data['free_sulfur_dioxide'],
            data['total_sulfur_dioxide'],
            data['density'], data['pH'],
            data['sulphates'],
            data['alcohol'],
            data['wine_type'],
        ]).reshape(1, -1)

        prediction = model.predict(input_features)[0]

        return jsonify({"prediction": float(prediction)})

    except (ValueError, TypeError, json.JSONDecodeError):
        return jsonify({"error": "Invalid input data"}), 400


if __name__ == '__main__':
    app.run(debug=True)
