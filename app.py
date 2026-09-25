from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('models/rf_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_fields = ['Property Type', 'County', 'BER Rating', 'Number of Bedrooms', 'Floor Area (m2)']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    input_df = pd.DataFrame([{
        'Property Type': data['Property Type'],
        'County': data['County'],
        'BER Rating': data['BER Rating'],
        'Number of Bedrooms': data['Number of Bedrooms'],
        'Floor Area (m2)': data['Floor Area (m2)']
    }])

    log_prediction = model_pipeline_predict = model.predict(input_df)[0]
    price_prediction = np.expm1(log_prediction)

    return jsonify({
        'predicted_price': round(float(price_prediction), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)