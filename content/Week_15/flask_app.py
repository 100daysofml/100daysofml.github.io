
from flask import Flask, request, jsonify
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load model and metadata at startup
model = joblib.load('iris_model.pkl')
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': metadata['model_type']})

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        # Get input data from request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Format response
        response = {
            'prediction': metadata['target_names'][prediction],
            'prediction_id': int(prediction),
            'probabilities': {
                name: float(prob) 
                for name, prob in zip(metadata['target_names'], probability)
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        features = np.array(data['features'])

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        response = {
            'predictions': [
                {
                    'prediction': metadata['target_names'][pred],
                    'prediction_id': int(pred),
                    'probabilities': {
                        name: float(prob)
                        for name, prob in zip(metadata['target_names'], probs)
                    }
                }
                for pred, probs in zip(predictions, probabilities)
            ]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model metadata"""
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
