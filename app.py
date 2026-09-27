from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
model = joblib.load('models/rf_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_fields = ['Property Type', 'County', 'BER Rating', 'Number of Bedrooms', 'Number of Bathrooms', 'Floor Area (m2)', 'Area_grouped']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    input_df = pd.DataFrame([{
        'Property Type': data['Property Type'],
        'County': data['County'],
        'BER Rating': data['BER Rating'],
        'Number of Bedrooms': data['Number of Bedrooms'],
        'Number of Bathrooms': data['Number of Bathrooms'],
        'Floor Area (m2)': data['Floor Area (m2)'],
        'Area_grouped': data['Area_grouped']
    }])

    log_prediction = model.predict(input_df)[0]
    price_prediction = np.expm1(log_prediction)

    return jsonify({
        'predicted_price': round(float(price_prediction), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)